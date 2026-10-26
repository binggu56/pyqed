GW and BSE
==========

PyQED provides a native dense molecular GW/BSE workflow in :mod:`pyqed.gw`.
The implementation is designed as a transparent reference backend for small
and medium molecules: it is useful for comparing against PySCF and MOLGW,
testing new approximations, and building neutral excited-state potential
energy surfaces.

The recommended workflow is:

.. code-block:: python

   mf = RHF(mol).run()
   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)

Mean-field, GW, and BSE have distinct roles:

* ``RHF`` builds the closed-shell reference, orbitals, and SCF total energy.
* ``GW`` computes quasiparticle energies and screening information.
* ``BSE`` computes neutral excitation energies from the GW/RPA reference.

``BSE`` and ``TDA`` expose ``run()`` for excitations and ``derivatives()`` for
analytic derivatives. They do not provide a quasiparticle ``kernel()``,
``sigma()``, or ``g0()``; use ``GW`` for self-energies and Green functions.
Passing an unfinished GW object raises an error. For direct BSE with
mean-field orbital gaps, pass the SCF object explicitly: ``BSE(mf).run()``.

For the analytic spectral derivative path:

.. code-block:: python

   gw = GW(mf, freq_int='exact', eta=0.).run()
   bse = BSE(gw, screening_energies='hf').run(nroots=2, tol=1e-10)
   gradient, hessian = bse.derivatives(order=2, state_ids=[0], modes=modes)

Here ``state_ids=[0]`` selects the first excited state. ``modes`` contains
Cartesian displacements per requested coordinate; supply dimensionless
mass- and frequency-weighted normal modes when derivatives with respect to
those coordinates are wanted. The combined derivative call automatically
releases its response workspace.

Default settings and PySCF comparison
-------------------------------------

Molecular ``GW(mf)`` defaults to imaginary-axis Padé G0W0, matching PySCF
2.14's 100 quadrature points, 18 fit points, nonlinear tolerance 1e-6 Ha,
and 100 iterations. ``freq_int='exact'`` explicitly selects spectral GW.
Ordinary PySCF SCF references automatically obtain the same MP2-fit auxiliary
basis policy as PySCF GW; supplied RI/CD factors are retained. Native PyQED
references require ``mol.build(eri='ri')`` or ``eri='cd'`` before SCF. Exchange follows
``vhf_df=False`` (reference exchange, exact AO exchange for PySCF DFT); use
``vhf_df=True`` explicitly for exchange from the GW auxiliary factors.

``BSE(gw)`` now builds static direct-RPA screening with **GW quasiparticle gaps**,
as PySCF BSE does. It retains the mean-field orbitals and all screening
transitions. The auxiliary-factor formulation follows
https://doi.org/10.1002/jcc.24688. The existing real restricted implementation
is an adaptation, not identical PySCF solver code. Full XY response, positive
metric/reference checks and the final physical residual audit remain.
Default full BSE requests ten singlet roots, 100 iterations, a 12*nroots
iterative space, and ``tol=1e-4`` Ha. The response solver checks both physical
XY norms and P/Q norms; the latter's squared bound is PySCF's default 1e-8.
PyQED can consequently converge more strictly or take different iterations.

Padé fit stability (16/18/20-point comparison at 1e-4 Ha), positive-weight
checks and explicit failures are preserved. Matching defaults does not imply
bitwise identical energies or root selection; PyQED does not reproduce the
known PySCF 2.14 continued-fraction evaluation defect.

For the earlier benchmark convention, use
``BSE(gw, screening_energies='hf')``. Only this explicit mode reuses the
HF-gap screening cached by GW. QP-gap screening requires finite GW energies
for **all screening orbitals**, even when ``orbital_window`` restricts BSE.
Missing QPs raise instead of being filled with HF energies. Changing the BSE
QP energies invalidates QP-gap screening. The archived anthracene/tetracene/
pentacene HF-screening benchmark results are not new-default predictions.

Molecular analytic-continuation G0W0
------------------------------------

For a real, gapped restricted RI/CD reference, the molecular AC path is:

.. code-block:: python

   gw = GW(mf, freq_int='ac', threads=8).run()
   bse = BSE(gw, threads=8)

This adapts the imaginary-axis RI-GW formulation of X. Ren et al.,
*Resolution-of-identity approach to Hartree–Fock, hybrid density functionals,
RPA, MP2 and GW with numeric atom-centered orbital basis functions*,
New J. Phys. **14**, 053020 (2012), https://doi.org/10.1088/1367-2630/14/5/053020,
and Zhu and Chan, JCTC **17**, 727–741 (2021),
https://doi.org/10.1021/acs.jctc.0c00704. Molecular and periodic AC share the
same two-pole least-squares fitting implementation. The dielectric construction
and mapped quadrature reuse the molecular contour machinery. All occupied-virtual
transitions and intermediate orbitals remain; pair projections are blocked
without forming four-index ERIs. Eight threads means one BLAS team of eight,
not eight orbital workers each starting eight BLAS threads. Imaginary-axis
screening uses symmetric BLAS rank updates, computing one triangle and
mirroring it after all transition blocks have been accumulated.

Only diagonal, one-shot, restricted direct-RPA (TDH) GW is supported. The
correlation self-energy is fitted on signed imaginary frequencies relative
to the midgap chemical potential. ``iw_cutoff`` limits fit evaluation frequencies,
not integration frequencies; ``nw`` controls the complete integration grid.
``orbs`` optionally selects spatial QP targets while retaining all intermediates.
Newton solves the fitted QP equation; ``linearized=True`` instead applies a
first-order correction. Neither prescription guarantees the same QP branch as
contour continuation. Failed or nonpositive-weight roots raise. Unrequested
energies are NaN. Select ``ac='pade'`` for Padé, or ``ac='twopole'`` for the
existing two-pole least-squares model. Padé is the molecular AC default.
The defaults match PySCF's settings: ``nw=100``, ``iw_cutoff=5.0``,
``pade_npts=18``, ``pade_step_ratio=2/3``, ``linearized=False``,
``tol=1e-6`` Hartree and ``maxiter=100``. Existing saved self-energies
retain their original integration grid when refitted.
Neither approximation has a general real-axis error guarantee. Check ``ac_info['fit_max_errors']``, quadrature
convergence and selected contour references before interpreting energies,
especially for deep core states. A small fit or QP residual does not bound
analytic-continuation error.

Padé adapts the reciprocal-difference continued-fraction interpolation of
H. J. Vidberg and J. W. Serene, *Solving the Eliashberg equations by means of
N-point Padé approximants*, J. Low Temp. Phys. **29**, 179–192 (1977),
https://doi.org/10.1007/BF00655090, to diagonal molecular self-energies.
The finite continued fraction is evaluated and differentiated analytically.
Point selection follows PySCF's nonuniform index-spacing prescription:
``pade_npts=18`` and ``pade_step_ratio=2/3``. The point count must be even,
at least two, and smaller than the available self-energy frequency count;
invalid, duplicate or out-of-range selections raise before screening starts.
Singular reciprocal differences raise rather than switching approximations.
This is not noise regularization or a causality-constrained fit. Padé may
introduce spurious poles: compare multiple point counts and inspect QP weights.
``ac_info['held_out_max_errors']`` reports errors at unused imaginary-axis
points, which still do not bound real-axis errors.

To compare continuations on exactly the same data, without touching screening:

.. code-block:: python

   from pyqed.gw.ac import refit
   refit(gw, mf.mo_energy, ac='twopole')
   two_pole_energies = gw.e_qp.copy()
   refit(gw, mf.mo_energy, ac='pade', pade_npts=18)

``refit`` reuses ``ac_static`` (exchange minus reference XC) and preserves the
imaginary-axis samples and static BSE screening. For a standalone saved-data
comparison, ``benchmarks/tdi_ac_refit.py`` reads the original TDI checkpoint,
recovers the omitted static shifts from its saved converged QP equations,
verifies two-pole reproduction, and compares Padé point counts on the same data.
It never loads RI factors or calls screening. Its figures are regenerated from
the saved self-energy, and its JSON records the input checksum.

``ac_sigma``, ``ac_omega``, ``ac_static``, ``ac_coeff`` and ``ac_nodes`` retain
the imaginary-axis data, static shifts and fit. ``qp_weights`` and ``qp_residuals`` expose QP diagnostics. The zero-frequency
Cholesky factor is reused by BSE only with ``screening_energies='hf'``; AC does not require a Casida eigensolve.
``max_memory`` controls estimated block scratch, not total process memory.
The reproducible TDI benchmark in ``benchmarks/tdi_ac_benchmark.py`` reuses
verified RHF and RI checkpoints, records stage timings and produces a diagnostic
figure. Saved checkpoints avoid repeating preparation; report this separately
when comparing end-to-end runtimes. ``benchmarks/tdi_ac_window_bse.py``
then times seven full-BSE roots in a four-occupied/four-virtual excitation
window using the complete cached static RPA response. That small window is
a handoff/timing check, not a converged spectrum or a full-space cost model.
Pass ``--pade-refit /path/to/refit-folder --output /path/to/new-bse-folder``
to use saved ``fits.npz`` Padé energies, with provenance checks and eight
OpenMP contraction threads. The cached static screening uses HF gaps; this
benchmark changes only the QP energies in BSE and does not reproduce a
QP-gap-screened PySCF BSE. No dielectric or GW integration is repeated.

``benchmarks/tdi_full_bse.py`` runs the full 271-occupied/527-virtual TDI
singlet BSE using all 798 completed Padé QP records from the supplied PySCF
log. It verifies geometry, orbital numbering, QP residuals and local-versus-
PySCF HF energies before reusing the local RHF/RI checkpoint. It rebuilds
only the static dielectric using QP gaps, caches its Cholesky and screened
pairs, and uses eight OpenMP contraction threads. SCF, RI generation and
GW integration are not repeated. The log supplies QPs to 12 decimal places
in Hartree; local MO coefficients are retained, so this is not a bitwise
reproduction of the cluster calculation. The local harmonic-Davidson solve
uses a 192-vector space, seven roots, 100 iterations and a 1e-6 Hartree
residual threshold. This smaller workspace changes solver convergence,
not the excitation space or static screening model. Reports include stage
and operator timings, residuals and observed thread counts; the final plot
compares excitation energies with the supplied PySCF BSE log.


Contour-deformation G0W0
------------------------

For larger restricted molecules with native CD or RI integrals, select
``freq_int='contour_deformation'`` (aliases ``'contour'`` and ``'cd'``):

.. code-block:: python

   from pyqed.gw.gw import GW

   # mf is a converged real restricted reference with CD/RI factors.
   no = int((mf.mo_occ > 0).sum())
   gw = GW(mf, freq_int='contour_deformation', eta=1e-6).run(
       orbs=[no-1, no], nw=64, quadrature_scale=0.5, tol=1e-9)
   print(gw.e_qp[gw.orbital_indices])
   print(gw.qp_residuals[gw.orbital_indices])
   sigma_c, sigma_x = gw.sigma(2*(no-1), 2*(no-1), gw.e_qp[no-1])

``orbs`` uses zero-based spatial indices, whereas ``sigma`` retains the GW
driver's spin-orbital indexing. Omit ``orbs`` to solve all QP energies.
Unrequested energies, QP weights and residuals are NaN; all intermediate
occupied and virtual orbitals still contribute to screening and self-energy.
Here ``cd`` means contour deformation; ``mol.build(eri='cd')`` independently
selects Cholesky integral factors.

This is a molecular adaptation of the density-fitted contour formulation of
T. Zhu and G. K.-L. Chan, *All-Electron Gaussian-Based G0W0 for Valence and
Core Excitation Energies of Periodic Systems*, JCTC **17**, 727–741 (2021),
https://doi.org/10.1021/acs.jctc.0c00704. The auxiliary dielectric matrix uses
the full direct-RPA response:

.. math::

   \Pi_{PQ}(i\nu) = -4\sum_{ia}
       \frac{\Delta_{ia} F_{Pia}F_{Qia}}{\nu^2+\Delta_{ia}^2}.

Transition blocks accumulate one auxiliary matrix at a time. Cholesky solves
on the imaginary axis and general solves for real-frequency residues replace
complete Casida diagonalization for W evaluation when using direct residues.
The default dominant root selection still requires the full charge-RPA
spectrum, and default spectral residues reuse it. The implementation integrates on a mapped
Gauss–Legendre grid and adds the occupied/virtual poles inside the deformed
contour. Subtracting the zero-frequency screened interaction treats the
orbital-pole step analytically, stabilizing evaluation at mean-field orbital
energies. It uses no analytic continuation, pole truncation, or additional
RI rank approximation.

Direct-dielectric screening workspace scales quadratically in the auxiliary
count. Dominant selection additionally needs quadratic transition-space storage. MO factors still occupy cubic storage, and
the saved imaginary-axis matrix elements scale as selected-orbital count
times MO count times ``nw``. Deep core or high virtual QP solves can be
expensive because they require many real-frequency residue evaluations.
Real-frequency polarization contracts the real and imaginary weights
separately with the real RI factors, avoiding complex copies of transition
blocks. Residues use a complex-symmetric (not Hermitian) dielectric solve.
Memory-dependent transition blocks contain up to 4096 transitions. These
are algebraically equivalent evaluations with no pole or orbital truncation;
floating-point summation order can change. Quadrature, regulator, and QP
root convergence controls are unchanged.

Contour contraction OpenMP
~~~~~~~~~~~~~~~~~~~~~~~~~~

``GW(mf, freq_int='contour_deformation', threads=8).run(...)`` uses
OpenMP for complete auxiliary dielectric construction and matrix-free
screening/pair contractions, with compiled imaginary-axis quadrature.
This exactly reassociates the molecular Zhu--Chan adaptation above;
all transitions, intermediate orbitals, regulator and root-selection
controls remain. Native output tiles have fixed summation order. Real and
imaginary dielectric products remain separate, and complex symmetric
matrices are mirrored without Hermitian conjugation. The GMRES residual
gate and Lanczos error controls are unchanged.

Screening preparation uses auxiliary/transition tiles. For direct residues
with multiple requested QP orbitals, the same thread budget then runs
independent orbital solves sharing immutable prepared screening. Each worker
owns its counters and caches and uses one contraction/BLAS thread, avoiding
nested pools. The existing TDI independent-orbital helper is shared with this
public path. Root continuation and numerical controls are unchanged.
Cholesky factorization, preconditioning and GMRES/Lanczos retain their
existing numerical-library thread controls. Dielectric construction
keeps one OpenMP team alive across transition blocks and packs the shared
right-hand tile in parallel. Each output tile retains the same transition
block order. On macOS its worker-local products use Accelerate DGEMM, with
BLAS threads fixed to one; other platforms retain the generic row-product
kernel. This changes floating-point evaluation inside each small product,
so agreement with older builds is numerical rather than bitwise. Serial
large products retain optimized NumPy BLAS; small compiled
tasks use one worker. Consequently complete direct solves may scale only
modestly, especially for small molecules. Keep BLAS thread limits at one
when measuring contraction scaling.

Direct residue results (screened value and derivative) are reused only for
identical frequency, orbital, intermediate state, regulator and tolerance.
No interpolation or approximate frequency matching is used. The per-worker
LRU cache is bounded by 4096 entries and an estimated response-cache budget.
The pivoted complex-symmetric LAPACK solve is unchanged, but its workspace
query is cached and the additional condition-number estimate in the general
SciPy wrapper is skipped. Singular-pivot, finite-solution and original
dielectric backward-error checks remain; RHS-relative residuals are recorded.
This does not introduce a stronger forward-error guarantee near poles.
Prepared reference factors and orbital energies must remain immutable.

``benchmarks/benchmark_gw_residue_reuse.py`` compares a saved pre-change
``contour.py`` with the current implementation for two- and four-orbital
water/cc-pVDZ calculations. It checks QP energies, weights, self-energies,
derivatives and residuals and saves an automatically generated timing plot.

Parallel kernels require row-contiguous real double-precision transition
factors. ``max_memory`` limits estimated temporary contraction storage and
may reduce observed teams and independent orbital workers; it does not cap RSS, factors, caches or mandatory
direct auxiliary matrices, and a minimum serial workspace remains.
``gw.threading_info`` and ``gw._contour.threading_info`` record requested
threads, maximum observed teams, per-kernel teams and calls. Reusing prepared
screening allows changing the contraction thread count; reference,
quadrature and screening controls must still match.

``PYQED_CONTOUR_OPENMP=required`` requires the extension and OpenMP at
build time; ``0`` builds its serial path. Missing OpenMP rejects requests
for multiple threads. ``benchmarks/contour_thread_scaling.py`` benchmarks
HOMO/LUMO QP roots with full screening, NumPy references, warm-up, repeated
timings, residual/weight/energy checks, profiles and diagnostic figures.
Fixed nw and eta establish thread agreement, not quadrature convergence.
``benchmarks/gw_openmp_phases.py`` separates inclusive dielectric,
screening-element, residue, and self-energy timings for water/cc-pVDZ with
complete screening and selected HOMO/LUMO roots. Nested times must not be
summed. Its ``--baseline`` option loads a saved contraction extension for
interleaved before/after comparisons in one process. Larger synthetic
factor matrices isolate kernel scaling; they are not molecular runtime
predictions. Quasiparticle weights, root residuals, self-energy values and
derivatives accompany the timing records.
When using the independent TDI ``--qp-workers`` driver described below,
keep inner contraction and BLAS thread counts at one to avoid nested fanout.

Experimental matrix-free real-frequency screening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GW(..., freq_int='contour').run(residue_solver='gmres',
screening_tol=1e-11, screening_maxiter=1000, screening_restart=128)``
replaces only the real-frequency residue linear solves. The imaginary-axis
quadrature and static BSE Cholesky factor still use direct screening. The
default ``residue_solver='direct'`` is unchanged.

The dielectric action is evaluated directly from all RI transition factors:

.. math::

   \epsilon(\omega)x=x-F\left[w(\omega)\odot(F^T x)\right].

This is an adaptation of restarted, left-preconditioned GMRES (Y. Saad and
M. H. Schultz, SIAM J. Sci. Stat. Comput. **7**, 856–869 (1986),
https://doi.org/10.1137/0907058) to the complete molecular direct-RPA
dielectric equation above. The retained zero-frequency Cholesky factor is
the preconditioner. There is no orbital, transition, or screening-pole
truncation, and no analytic continuation. ``screening_maxiter`` bounds total
inner iterations; ``screening_restart`` bounds the Krylov space. The true
unpreconditioned relative residual is recomputed before accepting a solve;
nonconvergence raises without a hidden direct fallback. Iteration counts,
maximum residual, and failures are recorded in ``gw._contour.screening_info``;
completed runs also copy these into ``gw.contour_info['screening_info']``.

Real and imaginary contractions are combined in contiguous auxiliary-row
blocks, including for disk-backed factors. Nearby-frequency residue solves
reuse the last converged vector for the same target/intermediate pair as a
GMRES initial guess. The equation and acceptance tolerance remain unchanged;
if a warm start fails, the same GMRES equation is retried from zero before
raising. The bounded cache retains at most 256 solution vectors.
``GW(..., mo_pair_factors=array)`` can reuse exact spatial MO RI/CD factors
without repeating the transformation. The caller must verify that the saved
coefficients, basis and auxiliary metric match the reference; shape checks
alone do not establish this. The TDI restart driver checks reference hashes
before loading its read-only factor maps.

The TDI window driver accepts ``--qp-workers N`` for independent orbital
G0W0 solves. Threads share the prepared full-screening arrays and factor
maps, while each owns its GW energies, contour statistics and warm guesses.
The existing serial GW kernel, quadrature, root continuation and residual
gates are used unchanged. Only the parent merges completed orbitals and
atomically replaces the checkpoint; completion order does not change the
orbital mapping. Limit BLAS/OpenMP to one thread per worker. Memory bandwidth
can limit scaling, so four concurrent solves are a starting point on the
32 GiB local TDI machine, rather than a guarantee of fourfold speedup.

Lanczos response reuse
~~~~~~~~~~~~~~~~~~~~~~

``residue_solver="lanczos"`` selects an experimental reusable inverse-Casida
response for contour residues. It is an adaptation of J. Laflamme Janssen,
B. Rousseau and M. Cote, Phys. Rev. B **91**, 125120 (2015),
https://doi.org/10.1103/PhysRevB.91.125120, and the full-frequency resolvent
idea of W. Gao et al., Phys. Rev. Lett. **132**, 126402 (2024),
https://doi.org/10.1103/PhysRevLett.132.126402. It is not an exact reproduction
of either implementation and does not use Gao et al.'s fitted polynomial
square root. For the current real, gapped, restricted direct-RPA reference,
let :math:`D` contain all occupied-virtual gaps and :math:`F` all transition
factors. Then

.. math::

   C=D^2+4\sqrt{D}F^TF\sqrt{D},\qquad K=C^{-1},
   \qquad b=2\sqrt{D}F^T r,

   r^T W_c(z)r=-b^T K(I-z^2K)^{-1}b.

Woodbury applies :math:`K` with the complete static dielectric Cholesky
factor. Two-pass reorthogonalized Lanczos chains are retained for each
needed target/intermediate pair, extended when necessary, and reused as the
QP continuation changes frequency. The cache has a bounded memory budget.
All screening transitions, intermediate orbitals, finite eta, imaginary
quadrature and static BSE screening are retained. There is no continuation
fit, screening-pole cutoff, or switch to TDA.

The spectral interval :math:`0\le K\le1/\min(D)^2` and Krylov residual give
separate value and derivative error estimates, with a relation-defect
allowance. The bounds apply in exact arithmetic; finite-precision input
factorization/roundoff errors require independent numerical validation.
Complete or invariant Krylov chains also check the original auxiliary
dielectric residual. The bounds can become conservative near poles.
Iteration or memory exhaustion raises without a hidden GMRES fallback;
``screening_maxiter`` limits chain length. This remains diagonal G0W0 for
real restricted TDH references only. Literature benchmark accuracy and
speedup guarantees do not transfer to this adaptation.

``benchmarks/benchmark_tdi_lanczos_response.py`` compares response values
and derivatives with current GMRES, recomputes all 16 QPs in the TDI 8/8
window, compares the seven lowest full-BSE excitations, and times one
current-GMRES QP solve. It reuses verified reference caches and writes its
own outputs separately from the convergence study.

The TDI window driver ``benchmarks/tdi_window_convergence.py`` defaults to
``--residue-solver lanczos`` following the 8/8 GW and full-BSE validation.
Direct and GMRES remain explicit options. This workflow default does not
change the general contour API's direct-solver default; larger windows still
require the same convergence and response-error checks.

``screening_info`` distinguishes ``response_chains_created``,
``response_build_seconds`` (including extensions),
``reused_response_seconds`` and ``response_cache_evictions``. A residue-call
average mixes expensive chain construction with cheap reuse and should not
be extrapolated directly to a larger QP target set.

``benchmarks/tdi_contour_pilot.py`` parses the supplied geometry/configuration
as data and runs a local selected-QP timing pilot with complete RI screening.
It reuses the existing TDI RHF checkpoint helpers, caches each screening
frequency, and checkpoints each completed QP. The default eight quadrature
points are for timing, not a reference convergence calculation. Eight BLAS
threads are used for preparation and screening right-hand sides; QP and BSE
contraction calls limit BLAS to one while using the requested OpenMP team.
Full-space BSE operator timings use HF gaps for uncomputed QPs and do not
constitute excitation energies or establish eigensolver convergence. Full
BSE Davidson uses complex trials, requiring two real contraction passes.
The memory setting controls allocations/storage decisions, not total RSS;
static screening arrays and Krylov caches must also fit the machine.

This does not guarantee faster convergence or self-energy accuracy near
real-frequency poles. Both screening and self-energy derivatives can be
ill-conditioned there, particularly at small ``eta``. The static
preconditioner can fail to converge in the excitation spectrum, and a small
linear residual alone is not an energy-error bound. Converge the linear
tolerance and compare residues, derivatives, QP energies, and BSE energies
against the direct solver before using this experimental option.

``benchmarks/benchmark_contour_matrix_free.py --output /private/tmp/contour-gmres``
benchmarks synthetic residues at several ranks and frequencies, records
iteration failures and traced allocation peaks, and compares molecular
direct and iterative GW. Water tests retain every QP orbital and compare
TDA/full BSE; benzene/6-31G tests one deep core orbital, the HOMO, and the
LUMO with the full screening/intermediate space. The small water
response diagonalizations are explicit dense validation references. Kernel
timings exclude the shared static factor setup; full molecular GW timings
include orbital transformation and imaginary screening. The script saves a
JSON report and reproducible diagnostic figure. Run with ``PYTHONPATH=.``
and one BLAS/OpenMP thread; these timings are not full TDI predictions.

In a local single-thread benchmark (three timed runs after warm-up),
benzene/6-31G with cc-pVDZ-JKFIT, 558 auxiliary functions, and core/HOMO/LUMO
QP targets took 16.74 s with direct screening and 9.49 s with GMRES. Their
QP energies agreed to the reported floating-point precision. Complete water
QP calculations at STO-3G and 6-31G agreed within 7e-16 Hartree; the three
lowest full-BSE excitations agreed within 1.6e-13 Hartree. These are reference
checks, not a general accuracy guarantee. Water/6-31G was slower with GMRES
(0.59 s versus 0.44 s). In synthetic rank-512 tests, low-frequency screening
was 2.36 times faster with roughly twelve times less traced workspace, but
two frequencies inside the excitation spectrum failed the 1000-iteration
limit. The direct default is therefore retained.

This implementation supports diagonal G0W0 with real, gapped restricted
references and TDH screening only. TDHF/TDDFT screening, evGW, qsGW,
off-diagonal self-energies, unrestricted and periodic formulations are not
implemented in this path. The existing positive-weight coupling-continuation
root prescription is retained, including its branch-selection limitations.
Converge ``nw`` (for example 32, 64, 128) and decrease ``eta`` independently;
the contour default is ``eta=1e-6`` Hartree (spectral default: ``1e-2``).
Agreement with the spectral equations is in the converged-quadrature,
zero-regulator limit. The finite-eta regularization differs from spectral
pole broadening. ``qp_residuals`` measures root convergence, not quadrature
error; ``info['contour']`` records the controls used.

BSE and BSE TDA reuse the contour driver's zero-frequency auxiliary Cholesky
factor when the factors, orbital coefficients and mean-field energies match.
Both full A/B actions and TDA avoid a screening eigensolve. The existing
static BSE kernel is evaluated through

.. math::

   R = \frac{I-\epsilon^{-1}(0)}{4},
   \qquad \sum_L \frac{M_{pqL}M_{rsL}}{\Omega_L}
        = \sum_{PQ} F_{Ppq}R_{PQ}F_{Qrs}.

Blocked Cholesky solves apply this metric to occupied-occupied and
occupied-virtual pair factors; no full orbital-pair/pole tensor is produced.
This is an exact reformulation of the existing static BSE kernel (Bruneval,
JCP **136**, 194107 (2012), https://doi.org/10.1063/1.4718428), with no new
screening truncation. It does not add a dynamical BSE kernel. The default
iterative excitation solver remains matrix-free; explicit ``low_rank=False``
still constructs and diagonalizes the BSE response matrix for small references.

.. code-block:: python

   from pyqed.gw.bse import BSE, TDA

   gw = GW(mf, freq_int='contour_deformation').run(nw=64)
   bse = BSE(gw).run(nroots=3, batch_columns=2)
   tda = TDA(gw).run(nroots=3, batch_columns=2)

BSE without an orbital window requires finite QP energies for all orbitals
and rejects a partial ``orbs`` calculation. A matching contour reference supplies static screening
without populating ``bse.e_rpa`` or ``bse._M``; these remain None, rather than
holding artificial screening poles. Changed reference orbitals or screening
energies invalidate reuse. Spectral screening remains available for other
references and screening methods.

Restricted excitation windows with full screening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BSE(gw, orbital_window=(no, nv))`` and ``TDA(gw, orbital_window=(no, nv))``
use the highest ``no`` occupied and lowest ``nv`` virtual reference orbitals.
Run contour GW for these target orbitals; unrequested QP energies may remain
NaN. Every orbital in the window must have a finite QP energy. The GW
screening and intermediate-state sums still use the complete reference;
the BSE static dielectric factor (or complete factorized spectral poles) is
reused without rebuilding polarization from the restricted orbitals.
``response.orbital_indices`` maps its amplitudes to the full reference.

This is an orbital-window projection of the existing static molecular BSE
kernel, inspired by the molecular workflow of F. Bruneval, S. M. Hamed and
J. B. Neaton, *A systematic benchmark of the ab initio Bethe-Salpeter equation
approach for low-lying optical excitations of small organic molecules*,
JCP **142**, 244101 (2015), https://doi.org/10.1063/1.4922489. The full-space
kernel submatrices are reproduced exactly, but omitted electron-hole
configurations introduce a convergence-controlled approximation to full
excitation energies. This does not truncate screening, use frozen-core GW,
apply a scissor shift, or add a dynamical BSE kernel. Require a matching
factorized contour or spectral GW reference. Scanner/overlap workflows for
changing orbital windows are not implemented; embed amplitudes using their
full-reference indices to compare fixed-geometry state subspaces.

``reuse_screening=True`` in contour ``GW.run`` reuses a prepared
``gw._contour`` if the complete reference, factors, quadrature, regulator,
linear controls and target columns match. Previously computed QP values
are retained while new targets are solved. An invalid cache raises. The
contour solver evaluates selected exchange diagonals in auxiliary blocks,
avoiding a full exchange matrix when only diagonal QP corrections are used.

``benchmarks/tdi_window_convergence.py`` parses the attached TDI input script
as data and runs nested frontier windows with its full reference basis,
RI settings, quadrature and convergence tolerances. The default windows
are 8/8, 12/12, 20/20, 32/32 and 48/48 occupied/virtual orbitals. It computes
seven target excitations plus additional roots to complete a near-degenerate
cluster at the seventh-state boundary, checks the target subspace overlap,
and reports convergence only after two successive
enlargements have maximum energy changes below 10 meV and minimum squared
principal overlaps above 0.95. These practical checks are not rigorous
complete-space error bounds. It saves RHF, partial QP checkpoints, response
amplitudes, provenance, a JSON history, and reproducible diagnostic figures
outside the repository. Full screening remains an expensive shared step.
The driver defaults to four independent frequency processes sharing
read-only disk-backed full MO/transition factors, with one BLAS/OpenMP
thread per process. Frequencies use the same complete direct-RPA equation
and quadrature as serial preparation. Contiguous selected-target factor
blocks retain every intermediate MO, avoiding sparse column reads across
the large full-MO factor file without dropping any self-energy terms.
Every auxiliary Cholesky factor and
target matrix-element block is cached atomically under the output directory.
``--resume`` checks the reference input/settings and reuses converged RHF,
completed screening frequencies and partial QP data; additional target
columns can reuse existing full frequency factorizations. The advanced
``ContourDeformation(..., prepare=False)`` plus
``set_screening_frequency`` collection path cannot be used for self-energy
evaluation or QP solves until all frequencies have been collected.

``benchmarks/benchmark_gw_tda_memory.py --molecule anthracene --integrals ri
--gw-only --freq-int contour_deformation --nw 64 --orbs 46 47 --eta 1e-6
--output /private/tmp/anthracene-contour`` reproduces the larger-molecule
frontier-orbital memory check and writes diagnostic figures and an RSS trace.
Use ``--compare-contour SPECTRAL_REPORT CONTOUR_REPORT...`` to plot saved
reports with identical geometry, basis, integral preparation and ``eta``.
On anthracene/6-31G with cc-pVDZ-JKFIT and ``eta=1e-6``, the HOMO/LUMO
maximum errors against spectral GW were 1.35e-10 and 2.87e-13 Hartree at
32 and 64 quadrature points. The 64-point frontier calculation peaked at
395 MiB traced GW allocations and 1.55 GiB process RSS including preparation;
MO-factor preparation set its sampled RSS peak. These are one-run memory
measurements, not a process-RAM guarantee or an all-orbital timing comparison.
The tests also compare all water/STO-3G QP energies with spectral GW and
frontier energies with PySCF contour GW using identical RI factors.
``benchmarks/benchmark_bse_static_screening.py`` compares static-auxiliary and
spectral screening for TDA and full BSE in separate processes. Both use the
same complete QP energies from a saved native RI GW report, isolating the
screening representation. End-to-end contour GW/BSE is tested separately on
water against dense spectral BSE, with guards against a hidden RPA rebuild.
On anthracene/6-31G with cc-pVDZ-JKFIT (146 MOs, 4653 transitions),
the three lowest TDA and full-BSE energies agreed within 1.4e-15 and
6.2e-14 Hartree, respectively; all residuals were below 1e-8. In separate
processes, peak RSS fell from 1.81 to 1.64 GiB, including RI/RHF preparation.
Screening-stage traced allocations fell from 572 to 395 MiB. These fixed-QP
measurements isolate screening and do not time a complete all-orbital GW run.
With blocked MO transformation and auxiliary-blocked response contractions,
the same static-screening benchmark reduced traced allocation peaks from
395 to 92 MiB (GW preparation), 68 to 35 MiB (TDA), and 338 to 82 MiB
(full BSE). The final run's process RSS was 1.56 GiB versus 1.64 GiB before;
resident memory varies with native allocation and mapped-page residency.
Excitations changed by less than 2.6e-15 Hartree. Use ``--compare-labels
Before After`` with the benchmark's ``--compare`` option to plot storage
revisions at fixed QP energies.

Exact charge screening and quasiparticle root tracking
------------------------------------------------------

Restricted TDH/Casida screening in ``GW`` now uses the complete spatial
charge block. With occupied-virtual gaps D and Coulomb pair matrix K, it
forms and diagonalizes

.. math::

   C = D^2 + 4 D^{1/2} K D^{1/2}.

This is an exact spin adaptation of the Casida formulation (Stratmann, Scuseria and Frisch,
*J. Chem. Phys.* **109**, 8218 (1998)), not a truncated screening-pole
approximation. The other three spin sectors have zero Coulomb coupling and
do not contribute to the GW self-energy. All charge poles are retained.
The diagonal gap matrix replaces a generic dense matrix-square-root solve.

Default TDH ``GW.rpa()`` returns spatial transition vectors, with
nocc_spatial*nvir_spatial rows; ``get_m_rpa`` applies their sqrt(2) spin
normalization and continues to return spin-indexed couplings. TDHF, TDA
screening and explicit non-Casida calls retain the full spin formulation.
The direct RPA correlation-energy helper still uses its original full-spin
reference implementation. This reduction supports real restricted orbitals.

``BSE(gw, screening_energies='hf')`` reuses the spatial screening poles and couplings when the
screening is TDH and its mean-field energies and MO coefficients match the
cached GW reference exactly. Changed-energy evGW screening is not reused
for mean-field BSE screening. Cached arrays are copied into the BSE object.
This avoids a second spatial RPA solve without changing the static kernel.

Dominant quasiparticle roots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restricted TDH spectral GW and contour GW now select the largest spectral
weight among resolved poles of the diagonal Dyson Green function. This uses
the quasiparticle definition discussed by A. Marie and P.-F. Loos,
JCTC **19**, 3943–3957 (2023), doi:10.1021/acs.jctc.3c00281; PyQED does not
implement the SRG approximation of that paper here. The full spectral
self-energy follows Bruneval, JCP **136**, 194107 (2012),
doi:10.1063/1.4718428. The finite secular equation is equivalent to a real
symmetric arrowhead eigenproblem. Positive residues make it monotone between
self-energy poles. Analytic upper bounds on root weights prune intervals
that cannot beat the current winner; remaining intervals use safeguarded
Newton solves. This has no fixed real-frequency search window.

For numerical resolution only, residues at most ``tol**2/(100*N)`` are
excluded from selection, where N is the original pole count. Their total
arrowhead perturbation norm is at most ``tol/10`` Ha, bounding eigenvalue
shifts but not individual weights inside nearly degenerate clusters. Every
original residue remains in self-energy and contour validation. This is a
finite-precision adaptation, not an exact-arithmetic guarantee on unresolved
peaks. ``qp_selection_info`` records interval counts, excluded-weight bounds,
the coupling-resolution energy bound, and states with selected weights below
0.5. A small maximum weight means spectral fragmentation, not necessarily a
root-selection failure. Tests compare against independent dense arrowhead
diagonalization, including shifted references and symmetry-noise couplings.

The previous local positive-weight continuation can track satellites or
change branches with eta. It remains only for explicit reference requests
``root_selection='continuation'``. The experimental Lanczos GW driver also
retains the historical branch policy and is not covered by this validation.
Earlier benchmarks below used that prescription and are historical.

Spectral GW orbital OpenMP
~~~~~~~~~~~~~~~~~~~~~~~~~~

``GW(mf, freq_int='exact', threads=4).run()`` selects dominant poles using
compiled orbital OpenMP. Contour defaults to complete spectral W evaluation
and refines the same selected root; ``residue_solver='direct'`` independently
evaluates W through dielectric solves, but dominant selection still requires
full charge-RPA screening. Explicit continuation with direct residues is the
only contour combination that avoids this diagonalization.

Factorized orbital couplings are formed in bounded batches, limited by
``max_memory`` and the requested thread count, without materializing the
complete orbital-pair/pole tensor. BLAS is held at one thread during these
transforms in the legacy path. Dominant-pole transforms use the requested
BLAS budget before OpenMP selection; the teams never run concurrently.
OpenMP orbital workers evaluate scalar pole sums without BLAS. ``threading_info``
reports OpenMP availability, requested and observed thread counts, and the
orbital batch size. Memory limits or small batches can reduce observed
parallelism. ``max_memory`` is a batch-storage budget, not a process RSS cap.
At least one orbital's residue array must fit in memory even when its size
exceeds that budget.

The GW extension is included in default extension builds. Set
``PYQED_GW_OPENMP=required`` when building to require OpenMP, or ``0`` to
build its serial compiled path. macOS runtime discovery reuses the existing
build helper and honors ``PYQED_OPENMP_PREFIX``. With the extension absent,
dominant selection raises with a build instruction; it does not silently
substitute the old local-branch solver. ``benchmarks/gw_thread_scaling.py`` validates against a profiled
Python reference, separately measures orbital OpenMP and BLAS scaling,
records observed workers and residuals, and writes a reproducible figure.

Root continuation is a behavior change separate from the exact screening
reduction: high virtual roots may differ from the former secant-selected
branches. Comparisons isolating charge reduction therefore use the same
continuation solver for both full-spin and spatial-charge screening.
``benchmarks/benchmark_gw_charge_screening.py`` performs that comparison
on pyrazine with shared CD/RHF input and writes reproducible figures.

On pyrazine/6-31G with CD threshold 1e-10 and one thread, three-run median
GW time falls from 30.03 s (full-spin reference with the same safeguarded
QP solver) to 2.31 s, a 13.0x speedup. Maximum QP and BSE differences
between representations are 2.65e-14 and 1.53e-15 Hartree respectively.
BSE screening preparation falls from 3.93 s to 0.002 s; the batched
Davidson solve itself remains about 10.7 s. Relative to the original
secant-selected branches, eight high virtual QP energies change materially;
the first five BSE excitations shift by at most 0.550 meV.

A separate optimized pyrazine/6-31G* run (one measurement) takes 7.79 s
for CD/RHF, 17.25 s for GW, 0.006 s for BSE screening reuse, and 45.54 s
for five batched-Davidson BSE roots: about 71 s total. Its maximum BSE
residual is 6.35e-10 Hartree. This larger-basis run has no full-spin timing
comparison. Reproducible reports, source snapshots and PNG/PDF figures
are in ``/private/tmp/gw_charge_pyrazine_631g_final`` and
``/private/tmp/gw_charge_pyrazine_631gstar_bracket``.

Molecular BSE/TDA OpenMP
------------------------

``BSE(gw, threads=8).run(nroots=5, batch_columns=8)`` and
``TDA(gw, threads=8).run(nroots=5, batch_columns=8)`` distribute fixed
auxiliary-factor tiles across OpenMP threads. This is independent of
``GW(..., threads=...)``. The BSE default is one thread; ``batch_columns``
controls trial-vector batching, not the thread count. Geometry scanners
retain the GW and BSE thread controls separately.

The default ``batch_columns='auto'`` chooses a cap of eight vectors,
reduced to fit estimated contraction scratch at the requested thread count.
It does not require one vector per thread: auxiliary-factor tiles supply
parallel work even for a single vector. Shape and available scratch determine
the memory cost, and more threads reduce the budget available per worker.
Explicit positive integers override automatic batching; ``None`` selects
scalar callbacks. ARPACK and dense reference paths disable automatic vector
batching. ``solver.batch_columns`` and ``info['batch_columns']`` record the
selected cap. No universal optimal batch width is claimed.
``benchmarks/bse_batch_scaling.py`` measures widths 1/2/4/8/16 and automatic
batching separately from thread count, with warm-up, repeated timings, root
and residual agreement checks, and a reproducible comparison figure.

The kernel exactly reassociates the static RI BSE equations of Bruneval,
J. Chem. Phys. 136, 194107 (2012), https://doi.org/10.1063/1.4718428.
Exchange and screening terms share intermediates, and full BSE evaluates
both A/B halves while streaming each factor tile. No transition-space
matrix, complete orbital-pair/pole coupling tensor, or four-virtual ERI
tensor is formed. Every existing screening factor is retained. This is
an algebraic implementation change, not a dynamical BSE approximation;
any RI/CD and explicitly selected orbital-window approximations remain.
The TDI saved-data driver ``benchmarks/tdi_bse_contractions.py`` compares
an archived kernel with the current one on identical full-orbital-dimension
trial blocks, then times the complete saved operator. ``--plot-only``
regenerates its figure from the JSON report. On this machine, an identical
128-auxiliary, four-vector, eight-thread comparison took 4.827 s for the
row-loop kernel and 1.377 s for BLAS (3.50x). A full 6430-auxiliary action
on four dense real trials in complex storage took 90.51 s with BLAS.
Fourteen one-entry seeds took 24.05 s with the old kernel and 0.621 s
with direct RI columns on that same 128-auxiliary slice. The complete
6430-auxiliary seed action took 70.99 s (eight observed OpenMP threads).
Use ``--seed-only`` to reproduce this comparison.
These are operator timings, not converged full-BSE wall times; an attempted
larger auxiliary-block grouping was slower and is not the selected path.

The existing Davidson/ARPACK physical residual and metric gates are unchanged,
and their existing convergence limitations still apply.

Multiple threads require real double-precision factors and matching
``FactorizedCouplings`` or ``StaticScreening`` storage. This supports static
screening reused from both spectral and contour GW. Complex trial vectors
are handled through their real and imaginary components. Dense reference
solves and unsupported screening storage reject multiple threads. The
one-thread path can use the existing NumPy contractions when the extension
or supported storage is absent. Compiled factors require positive,
row-contiguous storage; Fortran-order and reversed views use the NumPy
path at one thread and are rejected explicitly with multiple threads.
Geometry scanners preserve the BSE orbital window, screening choice,
broadening and independent GW/BSE thread settings.

Direct ``BSE(mf, threads=...)`` and ``TDA(mf, threads=...)`` with RI/CD
references retain newly built spectral screening as ``FactorizedCouplings``
as well. ``get_m_rpa(..., factorized=False)`` explicitly materializes the
orbital-pair/pole couplings for reference calculations; normal factorized
screening construction stores only the auxiliary-to-pole projection.

Workers use private matrix-product scratch. Large products call the SciPy
Cython BLAS ``dgemm`` entry point, with BLAS held to one thread inside the
OpenMP team; tiny products retain row loops. Exactly zero imaginary parts
and resonant halves skip their contractions.
Davidson one-entry seeds use direct RI columns instead of dense trial
matrix products, directly in the same compiled OpenMP team. These are exact
algebraic shortcuts, not sparsification by a
numerical threshold. No nonzero entries are discarded. Tile contributions
are reduced in a fixed
factor order independent of thread count. ``max_memory`` limits estimated
contraction scratch and can reduce observed parallelism; it does not cap
process RSS, existing screening caches, or solver subspaces, and at least
one trial batch must fit in memory. ``threading_info`` records requested
threads, the maximum observed OpenMP team size, tile size and action-call
count.
Screening preparation and projected eigensolves retain their existing
numerical-library controls.

The extension is included in default builds. ``PYQED_BSE_OPENMP=required``
requires OpenMP at build time; ``0`` builds compiled serial contractions.
macOS discovery honors ``PYQED_OPENMP_PREFIX`` through the shared build
helper. ``benchmarks/bse_thread_scaling.py`` compares NumPy, compiled serial,
and 2/4/8-thread actions and five-root solves, with warm-up, repeated timings,
root/action agreement checks, residuals, profiles and reproducible figures.
Its default static-dielectric case exercises the BSE representation reused
after contour GW, but uses spectral QP preparation and does not time a
contour-GW calculation.

The build hook detects a mismatched OpenMP runtime in cached macOS binaries
and rebuilds them even if their sources are timestamp-current. To rebuild
all extensions explicitly after changing compiler or runtime settings,
use ``--force``.
For example, from the repository root in the intended Python environment::

    PYQED_OPENMP_PREFIX=/opt/homebrew/opt/libomp \
    PYQED_GW_OPENMP=required PYQED_BSE_OPENMP=required \
    PYQED_CONTOUR_OPENMP=required python setup.py build_ext --inplace --force

The shared build hook pins the selected runtime's absolute library path,
preventing an earlier Conda RPATH from selecting an incompatible libomp.
Check the resulting libraries with ``otool -L`` and run
``tests/test_gw_dyson.py``, ``tests/test_bse_openmp.py`` and
``tests/test_contour_openmp.py`` with the standard BLAS limits. These tests
check observed teams and numerical agreement at 1/2/4/8 threads. A present
but unloadable extension fails the tests instead of being treated as an
optional missing module.

Basic Example
-------------

.. code-block:: python

   from pyqed.qchem import Molecule
   from pyqed.qchem.hf import RHF
   from pyqed.gw import GW, BSE, TDA

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       basis="sto-3g",
       unit="angstrom",
   )
   mol.build(eri="dense")

   mf = RHF(mol).run()
   gw = GW(mf, screening="TDH", eta=1e-3).run()

   bse = BSE(gw).run(nroots=3)
   tda = TDA(gw).run(nroots=3)

   print("SCF total energy:", mf.e_tot)
   print("GW quasiparticle energies:", gw.e_qp)
   print("BSE excitation energies:", bse.e)
   print("TDA excitation energies:", tda.e)

The :class:`~pyqed.gw.gw.GW` object stores quasiparticle energies in
``gw.e_qp``.  The older name ``gw.egw`` is kept as a compatibility alias.
For GW only, ``gw.e`` mirrors ``gw.e_qp``.  For BSE and TDA, ``bse.e`` and
``tda.e`` are excitation energies, so quasiparticle input energies live in
``bse.e_qp`` and ``tda.e_qp``.

GW Flavors
----------

The main entry point is :class:`pyqed.gw.gw.GW`.  It currently supports
restricted closed-shell references and dense/factorized molecular integrals.

Available methods include:

* ``GW(mf).run(method="g0w0")`` or ``GW(mf).g0w0()`` for one-shot GW.
* ``GW(mf).evgw(update_screening=False)`` for eigenvalue-only ``GnW0``.
* ``GW(mf).evgw(update_screening=True)`` for eigenvalue-only ``GnWn``.
* ``GW(mf).qsgw()`` for a dense quasiparticle self-consistent reference path.
* ``SCGW(mf).run()`` for an experimental dense imaginary-axis scGW prototype.

``GW.run()`` returns the GW object so that downstream code can pass it directly
to BSE.  It still behaves like the quasiparticle-energy array in common NumPy
contexts:

.. code-block:: python

   gw = GW(mf).run()

   qp = gw.e_qp
   qp_array = np.asarray(gw)
   homo = gw[nocc - 1]
   qp_ev = gw * 27.211386245988

Experimental scGW Prototype
---------------------------

``pyqed.gw.scgw.SCGW`` is a small finite-basis imaginary-axis prototype for
self-consistent GW.  It stores full matrix Green's functions, polarizabilities,
screened interactions, and correlation self-energies on a symmetric imaginary
frequency grid.  The default grid is a tangent-mapped Gauss-Legendre
quadrature over the full imaginary axis.  The older finite uniform grid remains
available with ``grid="linear"`` for debugging.  Two modes are available:

* ``scgw0``: update ``G`` and ``Sigma`` while keeping the initial screened
  interaction ``W0`` fixed.
* ``scgw``: update ``G``, ``P``, ``W``, and ``Sigma`` every macroiteration.

Both modes can optionally rebuild the bare-exchange part from the current
Green's-function density matrix.

.. code-block:: python

   from pyqed.gw.scgw import SCGW

   scgw0 = SCGW(mf, nfreq=17, wmax=20.0).scgw0(
       max_cycle=20,
       conv_tol=1e-6,
       damping=0.2,
   )

   scgw = SCGW(mf, nfreq=17, wmax=20.0).scgw(
       max_cycle=20,
       conv_tol=1e-6,
       damping=0.2,
   )

   print(scgw.converged)
   print(scgw.mu, scgw.nelec)
   print(scgw.e_qp)       # static imaginary-axis diagnostic estimate
   print(scgw.e_tot)      # Galitskii-Migdal total energy
   print(scgw.energy_components)
   print(scgw.G.shape)    # (nfreq, nso, nso)

The same functionality is also exposed through the normal GW driver:

.. code-block:: python

   from pyqed.gw.gw import GW

   gw0 = GW(mf).scgw0(nfreq=17, wmax=20.0, max_cycle=20)
   gw = GW(mf).scgw(nfreq=17, wmax=20.0, max_cycle=20)

   print(gw0.scgw_result.W0 is not None)
   print(gw.scgw_result.info["update_screening"])

When the mean-field object carries ``mol.eri_factors``/``mf.eri_factors``,
the prototype keeps ``P`` and ``W`` in the auxiliary factor space instead of
expanding the four-index ERI tensor.  Dense integrals are still supported for
small reference calculations.

For grid checks, use the convergence helper.  The default tangent-mapped
imaginary-frequency quadrature covers the infinite axis and is usually more
useful than the small uniform finite-cutoff grid for the convolution prototype.
The shifted Green's functions in the ``P`` and ``Sigma`` convolutions use the
large-frequency tail ``G(z) = z^{-1} I + z^{-2} H + O(z^{-3})`` outside the
explicit grid, avoiding the hard cutoff that made earlier finite grids overly
sensitive to ``wmax``.

.. code-block:: python

   from pyqed.gw.scgw import frequency_convergence

   rows = frequency_convergence(
       mf,
       nfreq_values=(7, 9, 11, 13),
       wmax=10.0,        # tangent-grid frequency scale
       method="scgw0",
       run_kwargs={"max_cycle": 10, "damping": 0.3},
   )

   for row in rows:
       print(
           row["nfreq"],
           row["e_tot"],
           row["delta_e_tot"],
           row["delta_qp_max"],
           row["grid_converged"],
       )

Validation status: MOLGW 3.4 provides G0W0, GnW0, GnWn/evGW, and QSGW
reference paths, and PyQED has smoke tests against those.  MOLGW's public
input parser does not expose the same fully interacting imaginary-axis scGW
loop implemented here.  For this prototype, the immediate validation criterion
is therefore internal stability with respect to ``nfreq``, frequency scale,
``density_nfreq``, damping, and fixed- versus updated-screening choices.  If
``delta_e_tot`` and ``delta_qp_max`` do not satisfy the requested tolerances
and set ``grid_converged=True``, the result should be treated as an
algorithm/debug diagnostic, not a converged molecular prediction.

This is not yet a production scGW implementation.  In particular, analytic
continuation and forces are still future work.  The current chemical-potential
control fixes the electron count with a tail-corrected Matsubara sum of the
interacting Green's function and expands the chemical-potential bracket when
the interacting density lies outside the static frontier-orbital window.

scGW Theory
-----------

Self-consistent GW solves Hedin's equations with the vertex set to one,
``Gamma = 1``.  In an orthonormal molecular-orbital basis, the central Dyson
equation is

.. math::

   G(i\omega_n) =
   \left[
     (\mu + i\omega_n) I - h_0 - \Sigma_x - \Sigma_c(i\omega_n)
   \right]^{-1}.

Here ``h0`` is the one-particle Hamiltonian with the mean-field potential
removed, ``Sigma_x`` is the static exchange self-energy, and ``Sigma_c`` is the
dynamic correlation self-energy.

The independent-particle polarizability is built from the interacting Green's
function:

.. math::

   P_{pq,rs}(i\nu_m)
   =
   -\frac{1}{\beta}
   \sum_n
   G_{pr}(i\omega_n + i\nu_m)
   G_{sq}(i\omega_n).

The screened Coulomb interaction follows a Dyson-like equation:

.. math::

   W(i\nu_m) = v + v P(i\nu_m) W(i\nu_m)
             = \left[1 - vP(i\nu_m)\right]^{-1} v.

The GW correlation self-energy is then

.. math::

   \Sigma^c_{pq}(i\omega_n)
   =
   -\frac{1}{\beta}
   \sum_m
   G_{rs}(i\omega_n - i\nu_m)
   W^c_{pr,qs}(i\nu_m),

where ``W_c = W - v``.  The total self-energy is

.. math::

   \Sigma(i\omega_n) = \Sigma_x + \Sigma_c(i\omega_n).

The chemical potential is adjusted to conserve particle number.  The density
matrix is obtained from the interacting Green's function using a high-frequency
tail correction:

.. math::

   \gamma =
   \frac{1}{2} I
   +
   \frac{1}{\beta}
   \sum_n
   \left[
     G(i\omega_n) - \frac{I}{i\omega_n}
   \right],

and ``mu`` is solved so that

.. math::

   N = \mathrm{Tr}\,\gamma.

The current PyQED prototype uses dense tensors and numerical interpolation on
finite imaginary-frequency grids.  It is therefore a reference implementation
for algorithm development, not yet a high-accuracy production scGW solver.

Galitskii-Migdal and Luttinger-Ward Energies
--------------------------------------------

For an electronic Hamiltonian

.. math::

   \hat H =
   \sum_{pq} h_{pq} c_p^\dagger c_q
   +
   \frac{1}{2}
   \sum_{pqrs}
   v_{pqrs}
   c_p^\dagger c_r^\dagger c_s c_q,

the one-particle density matrix is

.. math::

   \gamma_{pq} = \langle c_q^\dagger c_p \rangle.

The exact equation of motion for the Green's function gives the
Galitskii-Migdal interaction-energy identity

.. math::

   E_{\mathrm{int}}
   =
   \frac{1}{2\beta}
   \sum_n
   e^{i\omega_n 0^+}
   \mathrm{Tr}
   \left[
     \Sigma(i\omega_n) G(i\omega_n)
   \right].

Separating the self-energy into Hartree, exchange, and dynamic correlation
parts gives the practical molecular total energy used by ``SCGW``:

.. math::

   E_{\mathrm{GM}}
   =
   \mathrm{Tr}[h\gamma]
   + E_H[\gamma]
   + \frac{1}{2}\mathrm{Tr}[\Sigma_x \gamma]
   + \frac{1}{2\beta}
     \sum_n
     \mathrm{Tr}
     [
       \Sigma_c(i\omega_n)G(i\omega_n)
     ]
   + E_{\mathrm{nuc}}.

Here

.. math::

   E_H[\gamma]
   =
   \frac{1}{2}
   \sum_{pqrs}
   v_{pqrs}
   \gamma_{qp}\gamma_{sr}.

The factor ``1/2`` in the self-energy trace removes the double counting of
the two fermion lines attached to the interaction.

The Luttinger-Ward functional ``Phi[G]`` is defined so that

.. math::

   \Sigma = \frac{\delta \Phi[G]}{\delta G}.

For the GW skeleton functional, the correlation contribution satisfies

.. math::

   \Phi_c^{GW}[G]
   =
   \frac{1}{2\beta}
   \sum_n
   \mathrm{Tr}
   [
     \Sigma_c(i\omega_n)G(i\omega_n)
   ],

at self-consistency.  Therefore the stationary Luttinger-Ward internal energy
and the Galitskii-Migdal total energy are identical for the self-consistent GW
solution.  PyQED reports both ``e_tot_gm`` and ``e_tot_lw``; their difference
is a useful implementation and convergence diagnostic.

BSE and TDA
-----------

The preferred BSE API takes a completed GW object:

.. code-block:: python

   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)
   tda = TDA(gw).run(nroots=5)

Equivalently, use the convenience constructors on the GW object:

.. code-block:: python

   bse = gw.bse().run(nroots=5)
   tda = gw.tda().run(nroots=5)

``BSE`` solves the full Bethe-Salpeter eigenproblem and stores stacked
``X/Y`` amplitudes in ``bse.xy``.  The views ``bse.x`` and ``bse.y`` return
the excitation and de-excitation blocks.  ``TDA`` solves the Tamm-Dancoff
approximation and stores amplitudes in ``tda.x`` only.

For direct BSE calculations that follow the common MOLGW convention of using
HF/gKS orbital-energy differences instead of prior GW quasiparticle energies,
set ``use_qp=False``:

.. code-block:: python

   bse = BSE(gw).run(nroots=5, use_qp=False)
   tda = TDA(gw).run(nroots=5, use_qp=False)

Real response Davidson for molecular full BSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The matrix-free molecular full-BSE default is now ``eigensolver="response"``.
It uses one real orthonormal orbital-transition subspace for both X and Y.
Only new subspace columns are passed through the existing A/B action; the
small projected response problem uses the existing Casida reconstruction.
The physical, metric-normalized XY residual determines convergence. No full
transition matrix, TDA approximation, orbital window, or screening truncation
is introduced. ``max_space`` bounds the shared real subspace; ``batch_columns``
controls action width and ``None`` selects scalar callbacks.

This is an adaptation of R. E. Stratmann, G. E. Scuseria and M. J. Frisch,
"An efficient implementation of time-dependent density-functional theory
for the calculation of excitation energies of large molecules," *J. Chem.
Phys.* **109**, 8218–8224 (1998), https://doi.org/10.1063/1.477483, applied
to PyQED's existing static BSE kernel. It uses gap preconditioning, pivoted
QR, twice-projected corrections, thick restarts and a projected Casida solve.
It does not reproduce every solver detail of that reference or PySCF.
Real symmetric A/B and positive projected A+B and A-B are required. These
projected checks are not a global stability certificate or a guarantee that
all lower roots were found. Failures raise; the solver never silently switches
algorithms or returns unconverged roots. Complex-orbital and unstable-reference
support are not added. The general nonsymmetric solver remains an explicit
option for independent comparisons.

Nonsymmetric Davidson for molecular full BSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compiled nonsymmetric Davidson is available with ``eigensolver="davidson"``.
SciPy Arnoldi remains available with ``eigensolver="arpack"``:

.. code-block:: python

   bse = BSE(gw).run(
       nroots=5, low_rank=True, eigensolver="davidson",
       tol=1e-9, max_cycle=200, max_space=40, batch_columns=8,
   )
   print(bse.e, bse.info["residual_norms"], bse.info["metric_error"])

The same controls are accepted by ``BSE.solve_bse`` and the module-level
``solve_bse``. ``max_space`` is Davidson's subspace cap or ARPACK's ``ncv``.
``batch_columns`` applies the full A/B kernel to bounded blocks for Davidson;
it defaults to memory-aware automatic batching. Explicit ``None`` retains
scalar callbacks. Iterative controls require the
matrix-free path: use ``low_rank=True`` explicitly if the reference does not
have pair factors. Dense Casida, Hermitian TDA, periodic BSE, and GW screening
are unchanged. The existing real-orbital molecular kernel is used; accepting
complex trial vectors does not add complex-orbital BSE support.

The solver uses ``pyqed.linalg.davidson_nonsymmetric`` with positive-real
harmonic Ritz selection and signed orbital-energy gaps as an approximate
diagonal preconditioner (interaction contributions to the diagonal are
omitted). This is an adaptation of E. R. Davidson, *J. Comput. Phys.* **17**,
87–94 (1975), https://doi.org/10.1016/0021-9991(75)90065-0, with harmonic
extraction inspired by R. B. Morgan, *Linear Algebra Appl.* **154–156**,
289–309 (1991), https://doi.org/10.1016/0024-3795(91)90381-6. It is not a
structure-preserving BSE algorithm, a stability test, or a guarantee of
completeness of the lowest positive spectrum. There are no left eigenvectors
or biorthogonal iteration. Existing static screening and A/B couplings are
retained exactly; batching only changes contraction order.

Both iterative solvers reject partial convergence and a nonpositive selected
BSE metric. Nearly real roots may have complex eigenvector mixtures within
degenerate eigenspaces. For each root cluster the real and imaginary parts
are combined by SVD into an independent real basis, before BSE metric
orthonormalization. Clusters span at most the larger of 0.01*tol and a
100-machine-epsilon energy scale. A requested subset of a degenerate
multiplet can return any independent real basis of that subset. The full
physical residual check rejects invalid or excessively mixed real vectors.
After metric orthonormalization, every returned eigenpair must pass the
absolute residual threshold ``tol``; the metric error must be below 1e-8.
Normalization can amplify residuals near an instability, so a solver may
converge internally yet fail this final check. No silent fallback or tolerance
relaxation occurs. Positive roots elsewhere in an unstable spectrum may still
pass these selected-state checks; global stability must be established
separately. Failures leave ``info["converged"]`` false and raise an exception.
Davidson's diagnostics are in ``info["davidson"]``; ``operator_columns`` counts
all applied columns, including final validation.

``benchmarks/benchmark_bse_nonsymmetric.py --output /private/tmp/bse-comparison``
compares both solvers, including optional batching, against small dense
references. It uses synthetic symmetric factor kernels with the production
screened molecular actions, not ab initio molecules. Set the four documented
BLAS/OpenMP thread limits to one and run with ``PYTHONPATH=.``. Compilation and
warmup are excluded, three shuffled repeats are timed, and the report includes
post-normalization residuals, metric errors, operator counts, and reproducible
PNG/PDF figures. Use ``--plot-only`` to regenerate figures from the report.

A single-thread run on 23 September 2026 (three-repeat medians, five roots,
absolute normalized residual tolerance 1e-9) gave:

.. list-table:: Synthetic factorized full-BSE kernels
   :header-rows: 1

   * - Dimension
     - ARPACK (s)
     - Davidson (s)
     - Batched Davidson (s)
   * - 128
     - 0.158
     - 0.0259
     - 0.00968
   * - 384
     - 0.346
     - 0.0553
     - 0.0199

Operator-column counts, including validation, were 304 versus 40 and 564
versus 56 for ARPACK versus either Davidson action. These synthetic kernels
have useful gap preconditioning; the gains do not establish performance for
real molecules or weakly stable references. ARPACK was the default at that stage.
Reports and figures are saved under ``/private/tmp/bse_nonsymmetric_integration``.

Real-molecule qualification (23 September 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``benchmarks/benchmark_bse_molecules.py`` runs RHF/G0W0/BSE on LiH, HF, H2O,
NH3, N2 (1.098 Angstrom), and stretched N2 (1.80 Angstrom), all cc-pVDZ.
It uses PySCF RHF with explicit AO-to-MO transformation and PyQED exact-frequency
G0W0 with TDH screening and eta=0.001 Hartree. All six references converged
without QP Newton fallback; their dense A+B and A-B matrices are positive
definite. Five lowest positive roots are checked against independent dense
A/B spectra at an absolute normalized residual threshold of 1e-9 Hartree.
Each method has one warmup and three timed runs; the iteration cap is 200.

.. list-table:: Successful attempts, including warmup (out of four)
   :header-rows: 1

   * - Molecule
     - BSE dimension
     - ARPACK
     - Davidson (scalar and batched)
   * - LiH
     - 68
     - 4
     - 0
   * - HF
     - 140
     - 2
     - 0
   * - H2O
     - 190
     - 4
     - 4
   * - NH3
     - 240
     - 3
     - 4
   * - N2
     - 294
     - 2
     - 0
   * - Stretched N2
     - 294
     - 0
     - 0

In this pre-repair run Davidson failed its projected nonsymmetric eigensolve
on the four diatomic cases. The QR repair below resolves those failures. ARPACK failures are incomplete
convergence of its oversampled 14-root search. Neither failure is counted as
a valid time-to-solution. H2O medians are 1.990 s (ARPACK), 0.163 s (Davidson),
and 0.0533 s (batched Davidson). NH3 Davidson medians are 0.239 and 0.0764 s;
ARPACK's warmup failure prevents an all-attempts reliability claim.
Successful Davidson roots differ from dense references by less than
9e-12 Hartree, with normalized residuals below 7.6e-10 Hartree.

These pre-repair results ruled out inferring general robustness from synthetic
kernels alone; Davidson remained opt-in. ARPACK also needs care on clustered spectra.
The molecular cases, errors, geometries, settings, source hashes, logs, and
reproducible timing/residual and excitation-energy figures are recorded under
``/private/tmp/bse_real_molecules_20260923``. This is solver qualification,
not a claim of quantitative experimental accuracy or RHF orbital stability.

QR extraction and degenerate-state repair
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The harmonic projected failure was isolated to Accelerate's generalized QZ
solve on a finite, moderately conditioned LiH pencil after restart. The
solver now factors the shifted trial action by Householder QR and solves the
equivalent reciprocal ordinary eigenproblem using a triangular solve.
This retains Morgan's harmonic Ritz condition without forming normal equations
or a full physical matrix; see ``docs/nonsymmetric_davidson.md`` for the
formulation, diagnostics, and limitations. No tolerance or subspace-cap
relaxation is used.

The QR repair also exposed a conversion issue: degenerate real BSE states can
be returned as complex linear combinations rather than individually phase-real
vectors. The real-basis SVD conversion described above resolves it, including
when the requested root count cuts through a degenerate multiplet. Final
physical residual and positive-metric checks remain mandatory.

All 37 focused tests pass, including real/complex rotated degenerate blocks
that reproduce the old QZ failure and degenerate complex-mixture conversion.
The same six molecular cases now pass all four attempts with both Davidson
actions (24/24 each):

.. list-table:: Repaired solver medians, three timed repeats (seconds)
   :header-rows: 1

   * - Molecule
     - Davidson
     - Batched Davidson
   * - LiH
     - 0.0508
     - 0.0146
   * - HF
     - 0.0905
     - 0.0265
   * - H2O
     - 0.1414
     - 0.0456
   * - NH3
     - 0.2244
     - 0.0712
   * - N2
     - 0.2672
     - 0.0910
   * - Stretched N2
     - 0.2931
     - 0.1085

Maximum root error is 2.59e-10 Hartree and maximum normalized residual is
7.53e-10 Hartree, below the unchanged 1e-9 threshold. ARPACK's pass counts
remain 4, 2, 4, 3, 2, 0 out of four, under the same iteration cap. Its
accepted medians are 0.257 s for LiH and 1.662 s for H2O. These measurements
validate this molecular set, not universal interior-root completeness,
defective problems, Linux performance, or experimental accuracy. The default
solver was unchanged by the repair; the subsequent pyrazine qualification
below supports the switch to Davidson. Reports, figures, source snapshots and
reproduction instructions are in ``/private/tmp/bse_real_molecules_repaired``.

Pyrazine qualification and default selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The idealized planar D2h pyrazine geometry from
``benchmarks/benchmark_pyrazine_rhf.py`` was tested with STO-3G, RHF,
G0W0/TDH, five full-BSE roots, tolerance 1e-9 Hartree, and 200 iterations.
The independent dense reference has dimension 546; both A-B and A+B are
positive definite. There were no quasiparticle Newton fallback events.
Both Davidson variants passed one warmup and three timed repetitions in
13 iterations (95 operator columns). Median solver times were 1.184 s
for scalar Davidson and 0.527 s with ``batch_columns=8``. The maximum
root error was 2.04e-11 Hartree and normalized residual 8.48e-10 Hartree.
ARPACK failed all four attempts, returning only 10--13 of its 14 requested
Arnoldi roots at the iteration cap; partial results remain rejected.

Together with the six cc-pVDZ molecule checks above, this supported the earlier
nonsymmetric Davidson default. The current default is the real response solver
described above. ``BSE.run(eigensolver=None)`` selects
the path's default; explicit iterative solver controls require
``low_rank=True`` (or automatic selection through pair factors). Dense
reference dispatch is unchanged. Memory-aware automatic batching is now the
Davidson default; explicit ``None`` selects scalar callbacks. No fallback
or weaker residual gate was added. The pyrazine test qualifies this solver
case, not basis convergence or experimental excitation accuracy; a
cc-pVDZ pyrazine preparation was stopped in favor of this smaller complete
reference and supplies no validation result.

Reproduce with the repository's single-thread environment and ``PYTHONPATH=.``::

   python benchmarks/benchmark_bse_molecules.py --cases pyrazine \
       --basis sto-3g --output /private/tmp/bse_pyrazine_sto3g

Reports and reproducible comparison/energy figures are saved in that output
directory. Timing excludes RHF, GW, screening and dense-reference preparation.

Pyrazine / 6-31G with Cholesky integrals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The larger 62-orbital pyrazine case has a 1722-dimensional full-BSE matrix.
``benchmarks/benchmark_bse_molecules.py --cases pyrazine --basis 6-31g``
now accepts ``--integrals cd --cd-tol 1e-10`` for the PyQED CD/RHF path,
or ``--integrals exact`` for the independent PySCF RHF/exact-integral path.
CD retained 958 factors through GW/BSE without constructing dense molecular
ERIs on that path. Dense A/B matrices are validation only; the benchmark's
vectorized reference equations are checked against the original explicit
sums for both integral representations.

Both Davidson variants passed all four attempts in each workflow, using
five roots and tolerance 1e-9 Hartree. Timings are single-thread medians of
three repetitions, following one warmup:

.. list-table:: Pyrazine / 6-31G timings in seconds
   :header-rows: 1

   * - Integral workflow
     - GW preparation
     - Scalar Davidson
     - Batched Davidson (8 columns)
   * - Exact / PySCF RHF
     - 44.412
     - 17.784
     - 5.529
   * - CD / PyQED RHF
     - 21.740
     - 35.076
     - 11.063

Batching gives 3.17x speedup with CD and 3.22x with exact integrals.
CD reduces measured GW preparation by 2.04x, but its BSE contractions are
2.00x slower in the batched solve at this size. These preparation timings
include differing integral and RHF implementations; they are not an isolated
factorization algorithm comparison. The CD-versus-exact excitation shift
is at most 7.61e-6 Hartree (0.207 meV), including all RHF/GW numerical
differences. Both references are stable and neither uses a QP fallback.

ARPACK exceeded the 120-second wall limit in both workflows. The benchmark
now bounds each solver attempt with ``--solver-timeout`` and skips repeated
attempts after a failed warmup; timeout times are not converged solver
speedups. Solver timings exclude SCF, GW, screening and dense validation.
The reproducible reports/figures are under
``/private/tmp/bse_pyrazine_631g_comparison``; its ``compare.py`` combines
the ``bse_pyrazine_631g_exact`` and ``bse_pyrazine_631g_cd`` output directories.

Potential Energy Surfaces
-------------------------

For a neutral excited-state PES from BSE, use a consistent ground-state
reference at every geometry.  The practical default is:

.. math::

   E_0(R) = E_\mathrm{SCF}(R)

.. math::

   E_n(R) = E_\mathrm{SCF}(R) + \Omega_n^\mathrm{BSE}(R)

where ``mf.e_tot`` is the SCF total energy and ``bse.e[n]`` is the neutral BSE
excitation energy.

If an RPA-correlated ground-state reference is desired, use the same offset
for all excited states:

.. math::

   E_0(R) = E_\mathrm{SCF}(R) + E_c^\mathrm{RPA}(R)

.. math::

   E_n(R) = E_0(R) + \Omega_n^\mathrm{BSE}(R)

PyQED exposes this as:

.. code-block:: python

   gw = GW(mf).run()
   e0_rpa = gw.total_energy(method="rpa")
   bse = BSE(gw).run(nroots=3)
   excited_pes = e0_rpa + bse.e

The quasiparticle energies ``gw.e_qp`` should not be used directly as neutral
ground-state or excited-state PES energies; they correspond to charged
addition/removal quasiparticle levels.

Scanner Interface
-----------------

``BSE`` and ``TDA`` provide an ``as_scanner()`` helper for PES scans.  The
default scanner return value is ``[E0, E0 + Omega_1, ...]`` using the SCF
ground-state reference:

.. code-block:: python

   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=3)

   scanner = bse.as_scanner(nroots=3)
   energies = scanner(new_coords)

   e0 = energies[0]
   excited = energies[1:]

For excitation energies only:

.. code-block:: python

   omega_scanner = bse.as_scanner(nroots=3, energy="excitation")
   omega = omega_scanner(new_coords)

For an RPA-shifted PES:

.. code-block:: python

   rpa_scanner = bse.as_scanner(nroots=3, energy="rpa")
   energies = rpa_scanner(new_coords)

After each call, the scanner stores the latest objects as ``scanner.mf``,
``scanner.gw``, and ``scanner.bse``.

Wavefunction Overlaps
---------------------

BSE and TDA objects can compute overlaps between excitation vectors at
different geometries:

.. code-block:: python

   gw1 = GW(mf1).run()
   gw2 = GW(mf2).run()

   tda1 = TDA(gw1).run(nroots=3, return_vectors=True)
   tda2 = TDA(gw2).run(nroots=3, return_vectors=True)
   overlap_tda = tda1.wavefunction_overlap(tda2)

   bse1 = BSE(gw1).run(nroots=3, return_vectors=True)
   bse2 = BSE(gw2).run(nroots=3, return_vectors=True)
   overlap_bse = bse1.wavefunction_overlap(bse2)

This is useful for following states along a PES and diagnosing state flips.

Molecular and finite-periodic TDA/BSE overlaps share
``pyqed.qchem.singles_overlap.singlet_singles_overlap``. It contracts the
restricted singlet-CIS determinant algebra directly with the state columns,
without storing an excitation-pair metric. One occupied-block solve supplies
all row/column replacement factors; singular or ill-conditioned occupied
blocks (condition number above ``1e6``), or reference determinants whose
squares would underflow, use streamed direct determinants.
Frozen occupied orbitals, selected transition spaces, rectangular MO cross
metrics, complex phases, and ground-to-excited overlaps are retained.
Amplitudes passed to the helper are coefficients of normalized singlet CSFs;
the helper does not normalize states or unitarize their overlaps.

This is an exact algebraic adaptation of P.-O. Lowdin, *Phys. Rev.* **97**,
1474-1489 (1955), doi:10.1103/PhysRev.97.1474, using the matrix determinant
lemma and Schur complement. S. Lee, E. Kim, S. Lee, and C. H. Choi,
*JCTC* **15**, 882-891 (2019), doi:10.1021/acs.jctc.8b01049, motivates the
reuse of determinant factors, but the implementation does not reproduce
their spin-flip factorization or use their truncated Leibniz formula.
No small-displacement assumption is required. Exactness refers to the
represented restricted singlet-CIS states, not correlated GW wavefunctions.
The full-BSE expression remains the pseudo-overlap
:math:`X_L^\dagger S_{\mathrm{CIS}}X_R-Y_L^\dagger S_{\mathrm{CIS}}Y_R`;
this optimization adds no full-response wavefunction ansatz.

Integral Backends
-----------------

GW/BSE can use native dense integrals or factorized/RI inputs from the
mean-field reference:

.. code-block:: python

   mol.build(eri="ri", auxbasis="cc-pvdz-rifit")
   mf = RHF(mol).run(cholesky_jk=True)
   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)

When available, AO Cholesky or RI factors are transformed to MO pair factors.
This avoids storing the full four-index MO tensor in the GW self-energy and
low-rank BSE/TDA paths.  The dense reference solvers are still intended for
small and medium molecules.

Molecular GW consumes packed AO factors without expanding the complete AO
factor tensor. Default charge-screened quasiparticle energies and self-energies
contract spatial couplings directly, without constructing spin-expanded
couplings. With CD/RI, the default path stores auxiliary-to-pole coefficients
and forms only the requested orbital coupling blocks. It never stores the
full orbital-pair/pole coupling tensor. BSE TDA reuses these coefficients
through the static auxiliary metric and caches only its occupied-pair
contraction. This is exact reassociation of the existing RI spectral equations,
without additional rank or pole truncation. Explicit conversion of the
``FactorizedCouplings`` object to an array materializes the tensor and is
intended for small references. The explicit ``get_m_rpa`` helper still returns
spin couplings.
Auxiliary-to-pole arrays and explicitly requested spin spectral
couplings larger than one quarter of ``gw.max_memory`` (MB) use anonymous
temporary memory maps, released when the arrays are discarded. Set the
mean-field ``max_memory`` before constructing GW. Temporary storage follows
Python's temporary-directory configuration; use local disk for large runs.
This threshold does not cap process RSS or diagonalization workspace. The
complete charge spectrum still needs a dense matrix with quadratic storage
in the occupied-virtual pair count; no screening poles are truncated.
GW transforms packed AO factors in auxiliary blocks directly into MO-factor
storage. Arrays larger than one quarter of ``max_memory`` use anonymous
local temporary memory maps; smaller arrays remain in memory. This avoids
allocating the complete MO output on the Python heap, but mapped pages can
still contribute to RSS. AO factors remain owned by the mean-field reference. Native RI construction
keeps packed AO factors and spherical-transform work arrays larger than
32 MiB in anonymous temporary memory maps. Auxiliary transformation and
metric whitening use bounded pair blocks; Cholesky whitening uses triangular
solves. The RI metric, rank criterion, and integral values are unchanged.
Native integral generation can still allocate a full packed three-center
input, and mapped pages can remain resident during SCF. In the same
anthracene fixed-QP benchmark, these AO-storage changes reduced observed
whole-process peak RSS from 1.56 to 1.16 GiB. Full-BSE roots agreed within
1.4e-11 Hartree (residuals below 1e-8); this is a single-run measurement,
not a strict RSS cap. The post-SCF allocation peaks were essentially unchanged.

BSE direct and exchange contractions also block the auxiliary index, and
full-BSE A-action outputs are restricted to the current trial-vector batch.
These changes preserve every factor and excitation; no rank truncation or
additional physical approximation is introduced.

The charge Casida matrix is formed in place in LAPACK storage order and
consumed by diagonalization, avoiding separate Coulomb, symmetrization and
layout copies. Factorized spectral couplings are built in mode blocks sized
from the same memory threshold. These changes preserve the complete spectral
equations and do not introduce a screening approximation.

Native molecular ``pyqed.qchem.tddft.TDA`` uses Davidson response actions by
default. Its action transforms occupied-virtual, occupied-occupied and
virtual-virtual CD/RI factors in auxiliary blocks and consumes each block
immediately. Response preparation stores no complete MO pair factor tensor.
Block sizes account for AO transformation scratch and the trial-vector count.
This reduces persistent storage at the cost of repeated transformations;
TDA omits the unused B exchange action. Full TDHF retains both A and B.
Exchange contractions use explicit two-stage projections so NumPy cannot
select a costly direct three-operand contraction for larger trial blocks.
The molecular BSE TDA action also blocks auxiliary factors and screening
poles, including in scalar actions. Matching GW screening is shared without
copying its coupling tensor; iterative actions treat this storage as read-only.
``solver='dense'`` and ``get_ab()`` still explicitly construct response
matrices. The reproducible larger-molecule memory check is
``benchmarks/benchmark_gw_tda_memory.py`` (naphthalene, 6-31G).
Use ``--molecule anthracene`` or ``--molecule tetracene`` for larger checks,
and ``--integrals ri --auxbasis cc-pvdz-jkfit`` for native auxiliary-basis RI
instead of the default CD. ``--pyscf-ri`` explicitly selects an external
RI/RHF preparation reference; GW and both TDA actions still run in PyQED.
All calculation outputs are written to the requested ``--output`` directory.
``--tda-only`` profiles the response solve without rerunning GW/BSE; use
``--compare-tda OLD_CD NEW_CD OLD_RI NEW_RI`` to plot saved TDA allocation
measurements before and after a response change.
``--compare-gw BEFORE AFTER`` plots saved RI GW allocation peaks, full-run
process RSS, and orbital-by-orbital quasiparticle energy differences.
Reports also contain sampled resident memory by preparation, diagonalization,
coupling preparation, quasiparticle solve and BSE stages, with a corresponding
resident-memory diagnostic figure.

Matched PySCF 2.14 anthracene benchmark
---------------------------------------

``benchmarks/gw_bse_matched_scaling.py`` compares independent RI-G0W0 and
full singlet BSE calculations starting from a single saved RI-HF reference.
The default system is idealized planar anthracene, cc-pVDZ orbital basis and
cc-pVDZ-RI auxiliary basis: 246 orbitals, 47 occupied orbitals, 924 auxiliary
functions, and an 18,706-dimensional full BSE problem. No orbital window or
Tamm--Dancoff approximation is used. Five roots are requested.

Both codes use 100 imaginary-axis quadrature points, an 18-point Padé fit
with step ratio 2/3 and fitting cutoff 5 hartree, nonlinear quasiparticle
roots with tolerance 1e-7 hartree and at most 100 iterations. PyQED retains
its analytic-derivative Newton iteration; PySCF retains its secant iteration.
The BSE physical residual tolerance is 1e-6; PySCF's squared residual option
is set to 1e-12. Each final vector is independently checked in the full
X/Y equation after normalizing its positive indefinite norm to one.

The benchmark explicitly uses HF-gap direct-RPA screening in both GW and
BSE. This is a controlled W0 comparison: PySCF's BSE screening helper is
passed the shared HF energies instead of its default QP energies. Its BSE
diagonal still uses its independently computed QPs. This adaptation affects
only the benchmark process, not the installed solver. ``--common-qp`` is a
separate BSE-only control with identical saved QPs and W0; its timings must
not be mixed with end-to-end timings. ``--validate-inputs`` compares complete
MO factors, dielectric matrices, and screened blocks independently.

The formulation follows the diagonal imaginary-axis GW approach of Zhu and
Chan, JCTC 17, 727--741 (2021), doi:10.1021/acs.jctc.0c00704, with Padé
continuation. BSE uses the spin-adapted static-screening equations described
by Bruneval, Hamed and Neaton, J. Chem. Phys. 142, 244101 (2015),
doi:10.1063/1.4922489. Iterative convergence does not establish
Padé accuracy, basis convergence, or reliable deep-core quasiparticle
energies. Saved imaginary-axis data and fits can be diagnosed without
repeating GW using ``benchmarks/analyze_matched_gw_bse.py``.

Run from the repository root with ``PYTHONPATH=.`` and single-thread
``OPENBLAS_NUM_THREADS``, ``OMP_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS``,
and ``NUMEXPR_NUM_THREADS`` defaults. Explicit ``--threads`` sets the active
OpenMP/BLAS budget without nested pools. Use ``--prepare`` once, then
``--solver PyQED`` or ``--solver PySCF`` with ``--threads 1``, ``2``, ``4``,
or ``8`` and a numeric ``--repeat`` label. Each invocation is a fresh process;
run these sequentially. ``--profile`` runs are separate from timing results.
``--plot-only`` regenerates the comparison figure and JSON summary.

The official PySCF 2.14.0 ``pade_thiele_ndarray`` evaluator initializes its
continued-fraction tail with the final coefficient and then includes that
coefficient again in the backward loop. The diagnostic script reproduces
this with two samples of ``1 / (1 + z)`` at ``z = i, 2i``: the second input
is ``0.2 - 0.4i`` but the stock evaluator returns ``0.3 - 0.4i``. The main
benchmark deliberately retains the release evaluator. Corrected-evaluator
postprocessing is reported separately and never substituted silently into
the timing or accuracy comparison. Source provenance is checked against the
release wheel; see also the `PySCF continuation source
<https://pyscf.org/_modules/pyscf/gw/utils/ac_grid.html>`_.

Timings include each solver's MO transformation, GW, and BSE. They exclude
the shared RI-HF preparation, loading saved inputs, and independent validation.
All heavy inputs, wavefunctions, profiles, and raw logs should be kept outside
the repository (for example, ``/private/tmp/anthracene-gw-bse-matched``).

The 2026-10-13 Apple M2 Pro snapshot is archived in
``benchmarks/anthracene-gw-bse-pyscf/REPORT.rst`` with raw run JSON, numerical
audits, profiles, figures, and a reproduction script. At 1/2/4/8 threads,
PyQED took 292.0/169.3/100.0/73.7 s and PySCF took
252.9/134.5/70.1/47.1 s. These are single-run timings. The five excitation
energies span at most 2.40e-5 eV across both codes and all thread counts;
the identical-QP BSE control spans at most 7.13e-7 eV. Core QPs span up to
8.70 eV, so this is not a uniform validation of the GW quasiparticle spectrum.
GW orbital-pair projection consumes about 74% of PyQED's total time; the
four-thread profile identifies 7.53 s in repeated contiguous factor copies
inside GW, making invariant packing and buffer reuse the first optimization
target. BSE contractions and subspace work are the second target.

The subsequent projection optimization reuses consecutive target-factor views,
packs other selections once when the scratch budget permits, and reuses
projection buffers across frequencies. A bounded block-gather fallback handles
larger selections. The eight-thread anthracene rerun reduced projection time
from 54.73 to 35.74 s, GW from 63.02 to 45.32 s, and total time from 73.66 to
57.01 s. The matching PySCF rerun took 49.08 s. The imaginary-axis self-energy
was bit-for-bit unchanged; excitation energies changed by at most 4.71e-14 eV.
All physical residual gates passed. These are single-run timings; the data,
reproduction script, and diagnostic figure are in
``benchmarks/anthracene-gw-bse-projection/REPORT.rst``. Projection remains the
largest absolute cost; BSE is now the largest timing gap between the codes.

The subsequent BSE optimization batches X/Y factor products and complex trial
components, assembles outputs per auxiliary tile, and batches the final residual
checks. Compiled action calls fell from 39 to 20. Alternating BSE-only solves
with identical prepared data averaged 9.06 s before and 8.28 s after (8.6%
less time, two repeats). The eight-thread end-to-end rerun took 55.36 s for
PyQED and 47.82 s for PySCF; their BSE stages took 9.32 and 3.26 s. PyQED's
excitation energies changed by at most 3.56e-14 eV and all residual gates
passed. The static kernel and nonsymmetric Davidson solver are unchanged.
Larger scratch estimates account for the wider products. See
``benchmarks/anthracene-bse-optimized/REPORT.rst`` for raw results, caveats,
the paired control, profile, figure, and reproduction commands.

The real response Davidson follow-up closes most of that gap. On the same
eight-thread anthracene calculation, PyQED took 48.07 s versus PySCF's
46.93 s (2.4% apart); the BSE stages took 3.60 and 3.19 s, respectively.
PyQED's preceding BSE stage took 9.32 s. These are single-run timings.
The new real shared subspace converged in ten iterations, with estimated
solver workspace of 68 MB versus 313 MB for the general complex solver.
At the unchanged 1e-6 Ha residual threshold, excitation differences from the
old solver are below 0.714 microelectronvolt. Tightening both solvers to
1e-10 Ha reduced the maximum difference to 9.56e-11 eV. All independent
residual checks and 89 focused tests passed. The current response default,
its reference and stability restrictions are documented above; the matched
results, profile, figure and reproduction commands are archived in
``benchmarks/anthracene-bse-response/REPORT.rst``.

Validation Notes
----------------

The GW/BSE smoke tests currently cover:

* ``G0W0`` against PySCF exact-frequency GW.
* ``GnW0``, ``GnWn``, and ``qsGW`` against MOLGW reference data.
* dense and factorized integral consistency.
* dense and low-rank BSE/TDA consistency.
* same-geometry BSE/TDA overlap identities.

Related Examples
----------------

* ``examples/qchem/gw_qsgw.py``
* ``examples/qchem/gw_bse_ri.py``
* ``examples/qchem/gw/wavefunction_overlap.py``
* :doc:`qchem`
* :doc:`examples`

Experimental structure-preserving BSE Lanczos
---------------------------------------------

``BSE.run(eigensolver='lanczos', max_space=1000, max_cycle=200)`` selects
an optional real, stable full-BSE metric block Lanczos solver. The default
remains the preconditioned real response Davidson solver (``'response'``).
The Lanczos path reuses the compiled factorized BSE action; it constructs
neither the full BSE matrix nor its dense square root.

With ``M=A+B`` and ``K=A-B``, the implementation builds a block Krylov
basis for ``KM`` orthonormal in the ``M`` inner product. It reconstructs
both excitation and de-excitation amplitudes from the positive Ritz
frequencies. It uses low-gap unit vectors with seeded gap-weighted random perturbations
(block size equal to the requested roots), two-pass full metric
reorthogonalization, and no restart or residual preconditioner.
``max_space`` caps half-space basis columns. Exhausting the space or
iteration budget raises an error and retains convergence history in
``bse.info['lanczos']``; there is no silent Davidson fallback.
All returned eigenpairs pass the common fresh physical XY residual and
positive-metric checks. Only real symmetric stable operators are supported;
positive projected matrices do not certify global stability.

This is an eigenpair-oriented block adaptation of M. Shao, F. H. da Jornada,
L. Lin, C. Yang, J. Deslippe and S. G. Louie, "A structure preserving Lanczos
algorithm for computing the optical absorption spectrum," SIAM Journal on
Matrix Analysis and Applications **39**, 683–711 (2018),
https://doi.org/10.1137/16M1102641. It does not implement the paper's
optical quadrature or generalized averaged Gauss quadrature acceleration;
its spectrum approximation guarantees therefore do not carry over to
this eigenpair implementation. The separate GW screening Lanczos
resolvent implementation is unchanged.

Larger matched benchmark: tetracene at eight threads
----------------------------------------------------

The 2026-10-13 tetracene cc-pVDZ/cc-pVDZ-RI comparison uses 312 orbitals,
1176 auxiliary functions and a full BSE dimension of 30240. With the
current response Davidson default, one matched end-to-end run took
122.36 s for PyQED and 138.40 s for OpenMP-enabled PySCF 2.14. Both BSE
solvers converged in ten iterations with 89 trial vectors. Independent
GW calculations produced excitation differences of at most 13.30 micro-eV;
using identical QPs reduced the BSE difference to 1.73e-13 eV.

Three BSE-only runs per code with common QPs gave medians of 8.49 s for
PyQED and 7.35 s for PySCF. PySCF therefore retains a BSE throughput
advantage; the single end-to-end timing advantage comes from GW. PyQED GW
projection consumed 91.34 s, about 75% of the total solve time.

Matching low-energy BSE excitations does not validate all QP energies:
the largest QP discrepancy was 0.99 eV in a high virtual orbital and
0.42 eV among occupied orbitals. Imaginary-frequency self-energies agree
within 8.44e-17 Ha after aligning frequency conventions, locating this
remaining accuracy issue in the continuation/QP stage. Full conditions,
residual checks, repeat timings, figures and reproduction instructions
are in ``benchmarks/tetracene-gw-bse-pyscf/REPORT.rst``. This benchmark
does not use the experimental Lanczos eigensolver.

Continuation accuracy checks and explicit contour repair
--------------------------------------------------------

A small nonlinear QP residual establishes convergence to a fitted function,
not accuracy of its analytic continuation. Molecular Padé defaults to
``stability_tol=1e-4`` Ha. Both ``run`` and ``refit`` check the requested fit
against neighboring even point counts (normally 16 and 20
for an 18-point fit). A failed comparison, nonpositive-weight comparison
root, or change larger than ``stability_tol=1e-4`` Ha rejects the affected
orbitals. Their energies/weights/residuals become NaN and the solver remains
unconverged. Candidate energies and per-orbital changes remain in
``gw.ac_info['stability']`` for diagnosis. Passing this sensitivity heuristic
is not a real-axis accuracy certificate and does not establish accuracy at
the tighter nonlinear root tolerance.

When the check is enabled, the default ``on_unstable="raise"`` rejects
unstable roots instead of launching a
potentially very expensive calculation. To repair rejected states explicitly::

    gw = GW(mf, freq_int='ac', threads=8)
    gw.run(nw=100, stability_tol=1e-4, on_unstable='contour')

Explicit repair now recomputes **all requested states**, including Padé fits
that passed point-count sensitivity. It uses complete spectral screening and
largest-weight Dyson pole selection, then contour quadrature following
Zhu and Chan, JCTC 17, 727–741 (2021), doi:10.1021/acs.jctc.0c00704.
Subsequent ``gw.sigma`` calls use that contour model. No retained Padé root
is presumed accurate. All intermediate states and RPA poles remain in W.
This route requires the complete charge-RPA eigensystem; its storage is
quadratic in the number of transitions.

The contour regulator starts at ``eta=1e-6`` Ha independently of AC broadening.
Quadrature doubles and eta decreases tenfold independently until both changes
are below ``accuracy_tol=1e-5`` Ha. After ``max_refinements=3`` unsuccessful
rounds, repair raises and leaves requested energies unset. Diagnostics are
in ``gw.ac_info['contour_repair']``. Roots cannot leave their selected pole
interval during refinement. ``qp_weights < 0.01`` warns about fragmentation
even if this is the largest available weight. ``weak_root_orbitals`` lists
these states; numerical convergence does not establish that a single sharp
quasiparticle is an adequate description.

Direct real-frequency dielectric solves also check the linear residual
scaled by ``||epsilon||_F ||solution||_2 + ||rhs||_2`` and raise when
``screening_tol`` is not met; the RHS-relative residual is recorded too.
This backward-error gate allows floating-point cancellation at independent-
particle poles and supplements
the nonlinear Dyson residual and grid/regulator checks. A small linear
residual is not a bound on forward error at a poorly conditioned pole.
The repair also exits AC's multithreaded BLAS setting for its OpenMP
contractions, keeping inner BLAS at one thread.
Many rejected high-energy states can make this much slower than AC.
``pyqed.gw.ac.repair_continuation`` also repairs an existing object after a
rejected ``refit`` without repeating its imaginary-axis self-energy calculation.
``GW.from_checkpoint(path).refit(on_unstable="contour")`` provides the same
audited repair after restart. Parallel workers call the contour kernel
directly; they never re-enter the analytic-continuation driver.
It must construct the direct contour screening needed for those targets.

``stability_tol=None`` explicitly disables the guard and records the unchecked status.
Disabling the check does not stabilize the continuation or establish accuracy.

A failed base-fit QP solve is also retained as an invalid root and listed
in ``ac_info['root_failures']`` and the stability rejection list. It does not
abort before other orbitals can be diagnosed or silently retain a mean-field
energy. Explicit contour repair can process these failed roots alongside
point-count-sensitive roots.
QP root validation remains enabled.
``benchmarks/gw_bse_matched_scaling.py --unchecked-ac`` exposes that choice
and ``--correct-pyscf-pade`` explicitly applies and labels the reference
PySCF 2.14 evaluator correction described above. It does not modify the
installed PySCF package. Correcting that evaluator alone does not remove
ill-conditioning of the fits.

Saving BSE results and restarting
---------------------------------

``bse.save('bse.npz')`` exports available excitation results without pickle.
``excitation_energy`` and ``e_qp`` are in Hartree (``energy_unit='hartree'``),
with ``x`` and ``y`` amplitudes when available, orbital indices, the response
metric, and solver diagnostics in the JSON string ``info_json``. TDA omits
``y``. Saving requires excitation energies, but does not require convergence;
inspect the saved diagnostics before using unconverged results.

For a self-contained restart of a factorized static dielectric BSE problem::

    bse.checkpoint('bse-checkpoint')
    restarted = BSE.from_checkpoint('bse-checkpoint', threads=8)
    restarted.run(nroots=10, return_vectors=True)

The destination directory must not exist. The checkpoint stores QP/reference
energies, orbital indices and coefficients, RI factors, the static dielectric
Cholesky factor, any prepared occupied/occupied-virtual screening caches, and
available excitation results. Arrays are separate NPY files loaded read-only
with memory mapping; no pickle or external cache paths are used. Identical
factor objects are saved only once. Allow disk space for the RI factors and
screening caches. The manifest is written last, so interrupted saves without
a manifest cannot be loaded as complete checkpoints.

This restarts the eigensolve from the saved operator, without RHF, GW, or
static screening recomputation. It does not resume Davidson iteration history
or reconstruct molecular scanners, integrals, or a GW driver. Solver settings
are supplied to ``run`` again; loading defaults to one thread and optionally
accepts ``max_memory``. TDA checkpoints restore a TDA object. Dense and
spectral-coupling screening checkpoints are currently unsupported. Changing
QP energies after loading triggers the usual screening rebuild from saved
full-space factors. The storage layer introduces no method approximation.

Saving GW results and AC checkpoints
------------------------------------

GW and BSE use the same public naming: ``save(path)`` exports results,
``checkpoint(filename)`` writes restart state, and the class method
``from_checkpoint(filename)`` creates a solver from it. The latter is a
constructor, rather than a mutation of an existing solver.

``gw.save('gw.npz')`` stores available QP energies, weights, residuals,
orbital indices, AC self-energy samples/frequencies, static corrections,
and continuation coefficients/nodes. Energies use Hartree and
``metadata_json`` records convergence, method and diagnostics. Partial or
empty results can be exported after failure; inspect the metadata.

Molecular Padé ``gw.run()`` automatically saves a checkpoint after completing
imaginary-axis integration and computing the static correction, before any
Padé fitting or root solving. A unique ``gw-ac-*.chk`` HDF5 file is
created in the current working directory. Its absolute path is printed and
recorded in ``gw.ac_info['checkpoint_path']``. After a successful fit, the
automatic checkpoint is deleted and this field is removed. On failure it is
retained for recovery. Explicit ``checkpoint=path`` destinations are always
retained, including after success. Restored checkpoints are retained when
calling ``refit()``. This is a pre-fit snapshot:
restore it and call ``refit()`` to obtain QPs, even if the original fit failed::

    gw.chkfile = 'pyrazine-gw.chk'  # explicit destination is retained
    gw.run()
    # In another process, after failure:
    gw.chkfile = 'pyrazine-gw.chk'
    gw.run(restart=True, stability_tol=None)

Set ``gw.chkfile = None`` to disable saving. The default ``gw.chkfile`` is an
automatic file path; assigning a path makes it an explicit destination
that is always retained. ``GW.from_checkpoint(path)`` sets ``chkfile`` to the
restored HDF5 file. Arrays and JSON metadata are stored together in that file.
Per-run ``checkpoint=path`` and ``checkpoint=False`` override the attribute.
Explicit destinations must not exist; this is checked before integration. Automatic names are unique
and existing checkpoints are never overwritten. Checkpoints include RI factors
and can require substantial disk space. Two-pole runs do not save automatically,
but accept the same ``checkpoint`` option. Failed or interrupted integration
is not recoverable from this pre-fit checkpoint.

For molecular factorized AC GW::

    gw.checkpoint('gw.chk')
    gw = GW.from_checkpoint('gw.chk', threads=8)
    gw.refit(ac='pade')       # reuse saved self-energy; no integration
    # gw.refit(ac='twopole') # alternate continuation of the same samples
    bse = BSE(gw, threads=8)

The checkpoint includes MO RI factors, reference orbital data, available
static screening, sampled self-energy and fit/QP results. It uses a
pickle-free HDF5 file format with contiguous, uncompressed datasets. Large
arrays are restored as read-only memory maps directly into the file, so no
open HDF5 handle or separate array files are needed. The file is published
atomically only after writing completes, and existing files are never replaced.
The original SCF/molecule driver and active quadrature iteration are not restored;
``run()`` on a restored GW object raises with instructions to use ``refit()``
or construct a fresh GW driver. ``gw.run(restart=True)`` loads ``gw.chkfile``
and refits on the same object, replacing its current reference with the saved
reference. It accepts refit controls such as ``ac``, ``pade_npts`` and
``stability_tol``; integration controls such as ``nw`` and orbital overrides
are rejected. Saved continuation type, point count, spacing and linearization
are reused unless overridden. No screening or integral calculation is repeated
by restart. Explicit checkpoint files are retained; a failed automatic
checkpoint retried on its original object is removed after success. Refitting retains the existing AC method's
stability checks and approximations; it introduces no new approximation.

Default QP-gap BSE builds its own static screening from the restored QPs.
Explicit ``screening_energies='hf'`` can reuse the saved matching GW static
screening without rebuilding it. Contour-deformation, spectral GW and
contour-repaired AC checkpoints are currently unsupported; their available
QP results can still be exported with ``save()``. Checkpoint creation requires
saved AC samples and static corrections, and does not claim to resume an
interrupted integration.

Polynomial-square-root Lanczos GW
---------------------------------

``GW(mf, freq_int='lanczos', threads=4).run(orbs=[homo, lumo],
degree=16, steps=16, batch_columns='auto')`` selects a molecular adaptation of
W. Gao, Z. Tang, J. Zhao and J. R. Chelikowsky, "Efficient Full-Frequency GW
Calculations Using a Lanczos Method", Phys. Rev. Lett. **132**, 126402 (2024),
https://doi.org/10.1103/PhysRevLett.132.126402, particularly Eqs. 10–14.

This path uses the existing complete RI direct-RPA Casida action, approximates
its positive square root by a Chebyshev polynomial, and runs independent
Lanczos chains starting from each target/intermediate orbital-pair coupling.
The implementation batches these chains into BLAS matrix products and uses
two-pass classical Gram-Schmidt reorthogonalization with grouped projections
over the stored basis. Diagonal self-energies and derivatives are
computed from the small tridiagonal spectra, and the existing contour-GW
continuation solver selects positive-weight QP roots. No dense transition
matrix, frequency integration, Padé continuation, TDA, or intermediate-orbital
truncation is introduced. The existing ``residue_solver='lanczos'`` contour
option is a different algorithm and retains frequency quadrature.

This is an RI adaptation, not an exact reproduction of the paper's real-space
ISDF implementation. Only real, gapped, restricted, direct-RPA one-shot GW is
supported. ``degree`` and ``steps`` are independent approximation controls;
root convergence does not establish either approximation's convergence.
Repeat calculations with larger values before using the QPs as converged
results. All-electron core excitation energies can greatly increase the
polynomial degree required relative to the paper's pseudopotential examples.
The paper's accuracy and performance measurements do not transfer directly.

Automatic polynomial bounds use the minimum independent-particle gap squared
as a lower bound and an inflated largest Casida Ritz value as the upper
estimate. This upper estimate is not a certified enclosure; callers may pass
``spectral_bounds=(lower, upper)`` with verified bounds. ``lanczos_info``
records bounds, sampled scalar square-root approximation error, chain length,
number of Casida actions and their runtime. The sampled error is a diagnostic,
not a rigorous QP error bound. Memory limits control estimated chain scratch,
not the full RI factor storage or total resident memory.

Automatic batching caps independent chains at 256 columns, the number of
intermediate orbitals, and the estimated scratch allowance (half of
``max_memory``). Wider products amortize reading the RI factors; the useful
batch size depends on factor dimensions and memory as well as thread count.
An explicit positive integer overrides the automatic cap, subject to the
same memory allowance. This changes contraction scheduling, not the method
or its polynomial/chain convergence requirements.

The optional ``operator='casida'`` mode is a different spectral-quadrature
adaptation of the same RI direct-RPA spectral self-energy, not the paper's
polynomial-square-root algorithm. It forms Lanczos chains of ``C`` directly.
For each target/intermediate seed ``P``, its approximation is

.. math::

   P^T f(C)P \approx \|P\|^2 e_1^T f(T_m)e_1, \qquad
   f(x)=\frac{1}{\sqrt{x}\,[\omega-\epsilon_n+s_n(\sqrt{x}-i\eta)]}.

Here ``T_m`` is the projected Casida tridiagonal and ``s_n`` is positive
for occupied intermediate orbitals and negative for virtual orbitals.
The square root is evaluated on its small eigenspectrum. This avoids
``degree`` Casida actions inside every step, but has different finite-chain
convergence; ``degree`` is unused in this mode. The complete/invariant Krylov
limit gives the same spectral self-energy. Convergence of QP energies with
``steps`` must still be established, particularly near self-energy poles.
The default remains ``operator='sqrt'``. Both modes retain all RI transitions
and intermediate orbitals, with no dense full Casida matrix or TDA.

``benchmarks/profile_prl_lanczos.py`` compares both modes, Padé and contour
for anthracene/cc-pVDZ at four BLAS threads. It saves CPU profiles, QP results
and a reproducible diagnostic figure. ``--before-source`` accepts a saved
pre-optimization ``lanczos_gw.py`` for a matched polynomial baseline;
``--resume`` skips completed methods after checking the RHF orbital energies.

The local anthracene validation used 246 orbitals, 47 occupied orbitals,
9,353 transitions and 1,210 cc-pVDZ-JKFIT auxiliary functions. One profiled
run per method, excluding common RHF/RI setup, gave 17.7 s for Padé (100
points), 471.9 s for contour (128 points), and 489.4 s for the optimized
polynomial path (degree 128, 24 steps). The polynomial QPs differed from
contour by at most 2.69e-5 eV. Direct-Casida chains took 16.2, 44.1 and
139.5 s at 48, 96 and 192 steps, with maximum errors of 1.79e-4, 1.05e-7
and 5.38e-13 eV. Contour used one outer worker with four BLAS threads.
These are individual timings, not statistical speedup estimates.

Grouping the two-pass reorthogonalization projections reduced the same
48-step direct-Casida calculation from 24.5 to 16.2 s; its QPs changed by
less than 3e-14 eV. The optimized polynomial profile still spent 444.9 of
489.4 s in Casida actions. Thus the direct-Casida variant removes the main
nested-polynomial cost, but Padé remains faster at tight accuracy in this
benchmark. These measurements do not reproduce the PRL's ISDF benchmarks.

``benchmarks/benchmark_prl_lanczos.py`` benchmarks water/STO-3G with
cc-pVDZ-JKFIT, sharing the RHF orbitals and MO RI factors across Padé,
contour and polynomial Lanczos GW. Run it with ``PYTHONPATH=. python
benchmarks/benchmark_prl_lanczos.py --output /private/tmp/water-lanczos``;
``--plot-only`` regenerates the diagnostic figure from its saved report.
It uses four BLAS threads (one outer contour worker), three timing repeats,
and excludes common RHF/MO-RI setup. No core or transition truncation is used.

The local water validation gave median HOMO/LUMO runtimes of 0.024 s for
100-point Padé, 0.066 s for degree-128/10-step Lanczos, and 0.415 s for
128-point contour GW. Maximum deviations from contour were 3.9e-10 eV
and 7.6e-8 eV for Padé and Lanczos, respectively. Degree 256 reduced the
Lanczos deviation below 5e-13 eV, whereas degree 8 missed the HOMO by
0.160 eV. These tiny-system timings validate the implementation and its
convergence controls; they do not establish large-molecule scaling.

``benchmarks/tdi_prl_lanczos.py`` provides a four-thread TDI timing pilot,
full HOMO/LUMO chain calculation, a 64-point contour reference, saved-self-energy
Padé refitting and reproducible comparison plots. Padé refit-only timings are
explicitly separated from integration timings; they are not end-to-end GW
speedups. Cached data must belong to the same RHF/RI reference.

Full-spectrum reference validation
----------------------------------

``benchmarks/anthracene_contour_reference.py CHECKPOINT OUTPUT`` validates
saved anthracene PBE0 data by contour quadrature using every charge-RPA pole.
It reuses the exact spectral screening representation to evaluate the
screened interaction, instead of repeatedly solving the auxiliary dielectric
system at each real residue frequency. This is a validation implementation,
not a new production GW approximation: it retains every RPA pole, uses
bounded/disk-backed three-index couplings, and cross-checks selected values
and derivatives against direct dielectric solves. Quadrature and regulator
refinement are required. It uses the same restricted diagonal GW contour
formulation and root-continuation prescription documented above; numerical
convergence does not guarantee that a weak positive-weight Dyson pole is
the dominant quasiparticle peak.

The anthracene validation must distinguish quadrature convergence from
root-branch stability. A positive-weight root may move to a neighboring pole
as the real-axis regulator is reduced, even if imaginary-axis quadrature is
converged. Neither a small Dyson residual nor Padé point-count agreement
certifies its physical identity. ``repair_continuation`` rejects repairs that
fail its quadrature/regulator audit; do not feed such spectra into BSE as
validated results. The companion script
``benchmarks/plot_anthracene_contour_validation.py CHECKPOINT OUTPUT BASELINE_NPZ``
plots both successful and failed validation diagnostics. Its hybrid BSE is a
diagnostic comparison, not an assertion that the retained Padé states or
contour branch choices are accurate.

The earlier anthracene continuation audit above is superseded for current
root selection by ``benchmarks/anthracene_dominant_gw.py``. It exercises the
production dominant-pole solver and public checkpoint repair, compares
contour quadrature/regulator refinements and independently displaced initial
guesses, cross-checks real-frequency W against direct dielectric solves,
and recomputes BSE with matched QP screening.

Analytic relaxed molecular vibronic response
--------------------------------------------

Surface energy derivatives have their own callable API::

    F = excited.gradient(state_ids=[0, 1], modes=displacements)
    G = excited.hessian(state_ids=[0, 1], modes=displacements)

Here F has shape ``(nstates, nmodes)`` and G has shape
``(nstates, nmodes, nmodes)``. Results are also stored as
``energy_gradient`` and ``energy_hessian``. Both ``gradient()`` and
``hessian()`` default to the
Z-vector method with exact float64 disk-backed saved tensors. Use
``gradient(method="forward")`` to prepare reusable first-order integral,
orbital, screening and quasiparticle response for a following explicit forward
Hessian. ``hessian()`` computes analytic Z-vector second derivatives; afterward the gradient
is also available as ``energy_gradient``. A fresh gradient call clears
``energy_hessian`` to avoid exposing stale curvature.
With ``method="forward"``, first-order AO integral/factor, RHF, screening, self-energy, quasiparticle,
and BSE intermediates are cached on the BSE object. A following ``hessian(method="polarization")`` or ``hessian(method="mixed")``
with matching modes, reference contents, backend, orbital grouping, and GW/BSE
settings reuses their values and first derivatives, computing the missing
second-order terms. Mixed nuclear/frequency self-energy derivatives are needed
only for the Hessian and remain deferred until that call. Changes to arrays
in place are detected by content fingerprints. The cache retains one request
and can be released by setting ``_first_response_cache = None``. No public
reuse-status flag is needed.

When the all-directions response exceeds ``max_memory``, the engine evaluates
one analytic direction at a time and stores first-order arrays in temporary
files outside the repository. The explicit forward ``hessian(method='polarization')``
uses summed-direction curvatures. The opt-in ``hessian(method='mixed')``
implementation directly propagates
:math:`\partial_i\partial_j E` using the two cached first responses and one
mixed second-order block. The integral, RHF, GW, and BSE product rules retain
both cross terms, without constructing the two diagonal GW/BSE blocks. CD
integral projection emits only the requested pairs of directions. The existing
one-electron helper evaluates its small two-direction curvature block and
selects the cross entry. Diagonal Hessian passes run first and retain the
Hessian-only nuclear/frequency first responses for subsequent mixed pairs;
``gradient()`` still does not compute those Hessian-only terms.

The polarization implementation uses
:math:`H_{ij}=[H(e_i+e_j,e_i+e_j)-H_{ii}-H_{jj}]/2`.
Both paths retain all mixed entries and the same analytic response equations,
without finite differences, a diagonal approximation, or symmetry truncation.
The comparison script ``examples/qchem/gw/pyrazine_hessian_methods.py`` starts
both paths from independent copies of the same first-response cache and reports
wall times, sampled resident memory, pair timings, and Hessian differences.
It defaults to two bath modes; ``--mode-count 22`` selects the full bath.
Direct mixed response avoids subtracting diagonal curvatures, but carrying two
first directions can increase memory use and does not guarantee a speedup.
``examples/qchem/gw/pyrazine_response_profile.py`` accepts the same arguments
and writes exclusive and inclusive wall times for the second-order stages to
``stage_profile.json``. Exclusive times do not double-count nested stages.

CD curvature columns reuse the existing primitive-batched integral recurrence.
Directional Hessians additionally share each shell recurrence across selected
pairs. The batch size accounts for the returned columns and one accumulator
per worker, using at most one quarter of ``max_memory`` and at most 1 GiB for
those buffers, with a maximum of eight pairs. If two pairs do not fit, the
single-pair path is used. Completed columns are spilled to a temporary store
before electronic response is evaluated one pair at a time. Both direct and
polarization paths use this batching; their pair timings exclude the prefetched
integrals, while the total Hessian timing includes them. The stage profiler
records integral batches separately.
The recurrence and expanded-HRR batch buffers together are capped at 32 MiB
per worker; shells exceeding that cap or lacking proportional component weights
use the unbatched path. This cap covers the new batch buffers, not total process
memory. Only the current pair's electronic response tensors are retained; no
full AO ERI Hessian or batch of electronic pair-response tensors is formed. The full-spectrum
BSE surface and retained-subspace trace corrections share one state's coupling
buffer. Derivative matrices act on that state before the eigenbasis transform,
reducing the contraction from cubic to quadratic work in the excitation-space
dimension without truncating the intermediate-state sum.

A shared reference cache reuses CD/RI factor preparation, zeroth-order contractions,
screening square-root eigensystems, inverses, pole bases, and the exciton
eigensystem across directions and subsequent Hessian pairs. Hessian entries
also share the RHF orbital-response LU factorization once a Hessian
is requested; each second-order right-hand side uses triangular solves instead
of rebuilding and factoring the same matrix. Diagonal product-rule cross terms
are evaluated once and doubled. These changes preserve the response equations.
Factorized Coulomb/exchange builds, MO factor rotations, Casida products, and
screened BSE contractions use two-stage products. Their partial first responses
are retained in the same disk-backed cache, avoiding repeated partial transforms
in the second-order product rule.
Directional first responses remain separate. The same content fingerprint invalidates both
caches; large shared arrays use the persistent temporary store. Fixed operands
do not allocate zero response arrays or enter derivative contractions.
Contractions with exactly diagonal reference operands identify the contracted
indices instead of multiplying dense diagonal matrices; derivative operands
retain their full matrices. Exactly
diagonal reference matrices use elementwise square-root and inverse response
formulas while retaining all off-diagonal first and second derivatives. No
thresholding or diagonal approximation is applied to the response. Reference
screening matrices still scale with the full excitation space; the two first
responses and one second response must fit the working-memory estimate. Temporary files
are released when the cache and its detached copies are released.
CD directional curvature uses the existing compiled shell derivative
recurrence projected directly onto the fixed pivot columns, without a full
four-index derivative tensor. The fixed-pivot CD approximation is unchanged.
The default integral derivative backend follows the builtin AO basis. This
matters for general contractions such as cc-pVDZ: PySCF may use different
contracted AO functions spanning the same space, which cannot be aligned by
an AO permutation alone.


The shared analytic response engine is independent of the vibronic entry point.

``BSE.vibronic_couplings`` and ``TDA.vibronic_couplings`` provide first and
mixed second derivatives of a local effective electronic Hamiltonian::

    F, G = excited.vibronic_couplings(
        state_ids=[0, 1], modes=displacements, frame="relaxed")
    surface_gradients = excited.energy_gradient
    surface_hessians = excited.energy_hessian

Here ``displacements`` contains Cartesian ``dR/dQ`` in bohr, with shape
``(nmodes,natom,3)`` or either flattened orientation. State indices count
neutral excitations from zero; there is no implicit ground root. ``e_tot``
is ``E_RHF + excitation_energy``. The same relaxed RHF energy derivatives
are added to every diagonal. F/G have state indices first and one/two mode
indices last. Without modes their trailing axes are ``(natom,3)`` and
``(natom,3,natom,3)``. ``gradient`` and ``hessian`` always use flattened
Cartesian or supplied mode coordinates, with the selected state axis first.

This is analytic forward response: Gaussian integral derivatives, first and
second RHF CPHF, charge-RPA matrix-function response, implicit orbital-block
G0W0 Dyson-root response, and static BSE screening/kernel response are all
included. No displaced electronic solves enter the derivative algorithm.
Both ``screening_energies="hf"`` and ``"qp"`` are differentiated consistently.
The QP equation is differentiated at the selected zero-broadening root,
including its frequency chain rule. It assumes that root selection remains
on the same branch locally. It is not differentiable where the largest-weight
root changes discontinuously.

The implementation adapts the relaxed BSE@G0W0 formulation of J. Tölle,
M.-P. Kitsaras and P.-F. Loos, *Fully Analytic Nuclear Gradients for the
Bethe–Salpeter Equation*, J. Phys. Chem. Lett. **16**, 11134–11143 (2025),
https://doi.org/10.1021/acs.jpclett.5c02219. It differentiates PyQED's spectral
equations directly rather than reproducing that paper's Lagrangian/Z-vector
algorithm. Its second-order matrix response is an extension, not a published
accuracy guarantee from that work. Charge-RPA/GW and static BSE retain the
formulation of F. Bruneval, J. Chem. Phys. **136**, 194107 (2012),
https://doi.org/10.1063/1.4718428. First/second Gaussian integral response
reuses the Obara–Saika formulation, J. Chem. Phys. **84**, 3963 (1986),
https://doi.org/10.1063/1.450106. Fixed-pivot CD and moving-auxiliary RI
response reuse the existing factor-curvature implementation: Aquilante,
Lindh and Pedersen, JCP **129**, 034106 (2008),
https://doi.org/10.1063/1.2955755, and Dunlap, PCCP **2**, 2113 (2000),
https://doi.org/10.1039/B000027M.

Relaxed gradients versus local Hamiltonian curvature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an isolated electronic state, diagonal F equals its relaxed surface gradient. Diagonal G is the
second derivative of the matrix, not the eigenvalue Hessian. For an isolated
state I, the latter includes all other excitation states, including roots
not selected for output:

.. math::

   \partial_a\partial_b E_I = G_{II,ab}
    + \sum_{J\ne I}\frac{F_{IJ,a}F_{JI,b}+F_{IJ,b}F_{JI,a}}{E_I-E_J}.

``hessian`` includes this correction. At a degenerate excitation, individual
surface gradients/Hessians are generally undefined and returned as NaN, with an explicit flag in
``derivative_info``; matrix F/G remain meaningful in the chosen reference
basis. No inverse excitation gap is needed for F/G themselves.

The electronic frame is an explicit model choice. In TDA, reference exciton
amplitudes are frozen while orbital subspaces, matrix QP blocks and the kernel
relax. Full BSE uses the positive-frequency Hermitian Casida Hamiltonian

.. math::

   H_+ = \left[(A-B)^{1/2}(A+B)(A-B)^{1/2}\right]^{1/2}.

Its reference orthonormal eigenvectors are used to project derivatives.
These off-diagonal derivatives are effective-Hamiltonian coefficients;
they are not exact many-electron derivative couplings. In particular, the
moving orbital-block electronic frame has a transverse kinetic connection
that this interface does not supply. BSE full-response pseudo-overlaps also
remain an approximation, not ordinary many-electron overlaps.

RSH/CGLDR use
~~~~~~~~~~~~~

Automatic preparation supports this convention explicitly::

    dynamics = CGLDR(
        dvr, partition, state_ids=(0, 1), solver=excited,
        to_geometry=to_geometry, expansion_modes=displacements,
        derivative_frame="relaxed", electronic_cache=cache_path)
    dynamics.prepare_electronic_data()

The existing GW/BSE scanner now provides detached ``point`` objects, keeping
previous geometries and amplitudes intact for overlaps. Its ``point`` view
uses excited-only ``e_tot`` indexing even though its callable PES view can
include a ground root. The cache records the derivative convention and
rejects a mismatch. Default CGLDR still requests real-space fixed-state
coefficients; BSE rejects that convention rather than mislabeling relaxed
coefficients as fixed-state matrix elements.

This RSH/CGLDR use is the **Q-independent-link approximation**: the local
potential relaxes transversely, but transverse electronic kinetic connections
and transverse variation of inter-grid links are omitted. It is not an exact
projection of the moving BSE states. Converge the retained-state space and
local expansion range against independent displaced calculations before
using it for dynamics, especially near crossings.

Supported scope and checks
~~~~~~~~~~~~~~~~~~~~~~~~~~

The current implementation requires builtin, real closed-shell RHF,
``GW(..., freq_int="exact").run(method="g0w0")``, TDH screening and the
complete orbital space. It supports TDA and stable full BSE. DFT starting
points, evGW/qsGW, contour/AC/Lanczos GW derivatives, orbital windows,
complex orbitals and occupied/virtual gap closures are rejected. Degenerate
canonical orbitals and screening poles are supported by the block response
below. QP reference consistency and orbital-frame consistency are checked.
The reference QP root/residue must be scalar within each grouped orbital
subspace; an accidental HF degeneracy with a nonscalar GW correction needs
a full matrix Dyson reference and is rejected explicitly.

Exact-integral calculations use dense derivative ERIs. Native RI/CD retains
three-index factors throughout: RI requires an unscreened Cholesky auxiliary
metric, CD requires fixed pivots/rank, and integral screening must be zero.
No exact-ERI replacement of a fitted calculation occurs. Transition-space
response is dense; a conservative estimate is checked against ``max_memory``
in MB before allocation. Project a small number of transverse modes for
larger systems. This is a correctness-oriented implementation, not a
matrix-free production derivative engine.

``tests/test_bse_vibronic_response.py`` checks total gradients, surface
Hessians, off-diagonal F/G, mixed derivatives, both screening conventions,
TDA/full BSE, translational invariance, CD/RI response, and CGLDR integration
against small displaced references. The reproducible diagnostic script
``examples/qchem/bse_vibronic_validation.py`` writes numerical comparisons
and plots outside the repository.

Degenerate orbital and screening subspaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The response keeps a parallel gauge within each reference degenerate orbital
block and canonical response between blocks. ``degeneracy_tol`` (default
``1e-7`` Hartree) controls these blocks, separately for occupied and virtual
orbitals. No perturbation is used to split the reference multiplets. The
full first/second Fock block, including off-diagonal splitting operators,
is retained. Setting a small-gap denominator to zero while keeping only
individual orbital-energy derivatives would discard this splitting and is
not the implemented method.

For screening, the occupied–virtual gap is a matrix D and K is the bare
charge Coulomb matrix. Analytic matrix-function response uses

.. math::

   C=D^{1/2}(D+4K)D^{1/2},\qquad W_{\rm pole}=C^{1/2}.

First and mixed second square-root derivatives solve Sylvester equations
with sums of positive square-root eigenvalues, not differences between
screening poles. Inverse response uses the differentiated resolvent identity.
A reference-only pole-basis rotation diagonalizes the zero-order resolvent;
all off-diagonal pole-block response is retained. The self-energy resolvent
acts on the internal-orbital and screening axes separately. It never forms
a squared ``(n_internal*n_poles)`` Kronecker matrix.

This adapts higher-order matrix-function differentiation of N. J. Higham and
S. D. Relton, SIAM J. Matrix Anal. Appl. **35**, 1019–1037 (2014),
https://doi.org/10.1137/130945259, specialized to square roots and inverses.
It is not their general-purpose higher-order block algorithm.

A scalar diagonal GW approximation does not define invariant derivatives
under arbitrary rotations of degenerate orbitals. Consequently, within a
degenerate reference orbital block the implementation retains the **full
self-energy block** and its mixed frequency/nuclear response. If the
symmetry-protected root is w0 and its residue inverse is scalar z, the
Hermitian effective quasiparticle block has the expansion

.. math::

   H_a &= z^{-1}(E_a+\Sigma_a),\qquad z=1-\Sigma_\omega,\\
   H_{ab} &= z^{-1}\left[E_{ab}+\Sigma_{ab}
       +\tfrac12\{\Sigma_{\omega a},H_b\}
       +\tfrac12\{\Sigma_{\omega b},H_a\}
       +\tfrac12\Sigma_{\omega\omega}\{H_a,H_b\}\right].

The frequency derivatives are evaluated at w0. Here braces denote an
anticommutator; the reference frequency derivatives are scalar within the
symmetry multiplet. This is an adaptation of Hermitian effective-Hamiltonian
partitioning (P.-O. Löwdin, *A Note on the Quantum-Mechanical Perturbation
Theory*, J. Chem. Phys. **19**, 1396–1401 (1951),
https://doi.org/10.1063/1.1748067) to the spectral matrix Dyson equation.
It reproduces the selected block-Dyson eigenvalues through second order.
It is a **block extension** of diagonal G0W0: self-energy mixing between
distinct reference orbital blocks is still omitted. It is neither qsGW nor
a full matrix-GW calculation. Nonscalar reference roots/residues and branch
changes are not regularized into an apparently valid answer.

The BSE gap operator uses the resulting QP matrices, rather than keeping
only their diagonals. ``derivative_info`` records orbital blocks, reference
QP residuals, the tolerance and the effective-model convention. Include
whole electronic multiplets when constructing a local vibronic model;
individual state labels and diagonal matrix slopes within a degenerate
multiplet depend on the chosen basis. Directional energy slopes are the
eigenvalues of the corresponding full F block.

``tests/test_bse_subspace_response.py`` verifies arbitrary rotations within
occupied/virtual and repeated-pole subspaces, exact small matrix-Dyson
arrowhead references, and symmetry-breaking displacements for D3h H3+.
``examples/qchem/feco5_bse_subspace_response.py`` exercises unperturbed
D3h Fe(CO)5 with the complete minimal STO-3G orbital space and RI-STO-3G.
That example is a smoke test of degeneracy handling, not a spectroscopy
benchmark or a converged Fe(CO)5 vibronic parametrization.

Relation to LVC and QVC models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

F and G are precisely the Taylor coefficients of the chosen local effective
potential:

.. math::

   V(Q)=V_0+\sum_a F_aQ_a+\tfrac12\sum_{ab}G_{ab}Q_aQ_b.

F supplies the state slopes and interstate linear couplings of an LVC model.
G supplies the quadratic coefficients of a QVC model, including quadratic
Jahn–Teller coupling. At electronic degeneracies use the full F/G matrices,
not state-specific adiabatic Hessians. ``LVC`` and ``QVC`` accept them as
``linear_couplings`` and ``quadratic_couplings``::

    F, G = excited.vibronic_couplings(modes=dR_dQ, frame="relaxed")
    linear = LVC(E=excited.e_tot, omega=omega,
                 linear_couplings=F, normal_modes=dR_dQ)
    quadratic = QVC.from_lvc(linear, quadratic_couplings=G)
    V = quadratic.electronic_hamiltonian(Q, include_harmonic=False)

Coordinate normalization must match the model. For dimensionless oscillator
coordinates, ``omega`` is the oscillator energy in Hartree and ``dR_dQ``
includes the mass/frequency scaling. A common harmonic potential is usually
added separately in LVC. Because the returned G already contains the full
common RHF curvature, adding that harmonic term on top of the full G would
double-count it. Either evaluate the QVC with ``include_harmonic=False`` or
subtract the chosen common harmonic curvature times the electronic identity
from G before using ``include_harmonic=True``. The moving-frame and kinetic
approximations documented above still apply to these LVC/QVC coefficients.

Degenerate-state traces and a truncated electronic model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a complete retained multiplet (or a sum of complete multiplets), the
trace of its adiabatic energy is differentiable even when individual root
Hessians are not. ``subspace_gradient`` and ``subspace_hessian`` provide
these trace derivatives, with response through omitted excitation states.
They are NaN if the requested states cut a degenerate electronic multiplet.
For example, for a retained set P separated from its complement,

.. math::

   \partial_a\partial_b\sum_{I\in P}E_I
   =\sum_{I\in P}G_{II,ab}
   +\sum_{I\in P,K\notin P}
    \frac{F_{IK,a}F_{KI,b}+F_{IK,b}F_{KI,a}}{E_I-E_K}.

Thus the trace of projected G alone is not generally the retained energy
trace Hessian. Similarly, a QVC retaining only a few states does not
reproduce the full model's energies through second order unless the omitted
state response is negligible or explicitly downfolded. Raw F/G returned by
this interface are projected Hamiltonian derivatives, not automatically
downfolded coefficients. Increase the retained electronic space or apply a
separately justified effective-Hamiltonian construction before interpreting
such a small model quantitatively.

Minimal-basis Fe(CO)5 validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The D3h smoke calculation uses 68 STO-3G spatial orbitals, 960 occupied–virtual
transitions, RI-STO-3G Coulomb fitting, and eight verified low-lying TDA roots.
The z axis follows the axial CO bonds; the two test displacements translate
Fe along x/y in the equatorial plane. Nineteen degenerate orbital pairs are
retained without distorting the reference. RHF uses tight density/gradient
convergence. The existing safeguarded root-refinement helper tightens each
already selected GW branch inside its saved pole bracket; this refines root
residuals, not the original global spectral-weight selection bound.

The representative run has maximum QP block residual below 6e-13 Hartree,
x/y F-norm difference below 8e-13, and electronic Hermiticity error in G
below 2e-10. The retained energy-trace curvature is -0.4144965 Hartree/bohr^2;
central differences of independently displaced trace gradients at
+/-1e-4 bohr give -0.4151987 (0.17% disagreement at this finite step).
Energy second differences are much noisier because of subtraction of total
energies near -1800 Hartree. This is a finite-step smoke check, not an
extrapolated derivative error bound or a basis-converged spectroscopy result.

The same example also demonstrates that the projected eight-state G has a
large trace (about 1285.8) which is canceled by omitted-state response in the
energy-trace curvature. The projected eight-state QVC alone therefore does
not reproduce the displaced eight-state energies through second order.
This is the electronic-truncation issue described above; no downfolding is
silently added to raw F/G or to the active RSH/CGLDR potential.

Reproduce the checks with::

    PYTHONPATH=. python examples/qchem/feco5_bse_subspace_response.py
    PYTHONPATH=. python examples/qchem/feco5_bse_subspace_validation.py --gradient-check
    PYTHONPATH=. python examples/qchem/bse_degenerate_validation.py

Apply the repository's local BLAS/OpenMP limits before running. The first
script uses an explicitly identified transition-space validation diagonalization
and retains factors; it never constructs a four-index molecular ERI.
Outputs and reproducible figures default to ``/private/tmp``. The small D3h
H3+ example compares both LVC and QVC directly with displaced full-BSE spectra,
including the Jahn–Teller splitting of a complete doublet.

State-specific GW--BSE Z-vector gradients
----------------------------------------

``excited.gradient(method='adjoint', state_ids=[0], modes=None)`` computes
relaxed total-state gradients with a geometry-local reverse graph and one
RHF transpose solve per selected state. Install the optional ``pyqed[torch]``
extra. Tensor operations run on CPU in float64. This is a reverse-mode
adaptation of Tölle, Kitsaras and Loos, JPCL **16**, 11134--11143 (2025),
doi:10.1021/acs.jpclett.5c02219, rather than a reproduction of their explicit
Lagrangian. The orbital solve reuses ``OrbitalResponse`` and the RHF CPHF
operator; its adjoint organization follows Furche and Ahlrichs, JCP **117**,
7433 (2002), doi:10.1063/1.1508368. Matrix-square-root pullbacks solve
Sylvester equations, so repeated screening poles do not require eigenvector
derivatives. Quasiparticle pullbacks include the implicit Dyson residue;
HF-gap and QP-gap static screening are both differentiated.

The current adjoint implementation requires real builtin closed-shell RHF,
exact spectral G0W0, TDH screening, complete orbital spaces, and nondegenerate
HF orbital energies. It supports TDA and full BSE and the same dense or
fixed-rank CD / unscreened Cholesky RI integral models. Degenerate HF blocks
are explicitly rejected; use ``method='forward'`` for those. Individual
excitation gradients are NaN at electronic degeneracies. The graph and RHF
operator are dense and subject to an approximate memory gate; this is not a
matrix-free electronic adjoint. Saved reverse tensors are disk-backed by
default; individual dense operations still need working RAM. Nuclear integral derivatives
stream one direction at a time. Factor derivatives currently reuse the
existing analytic factor helper rather than a direct factor-adjoint shell
contraction. This remaining cost can limit the speedup.

The default is ``method='adjoint'``. Explicit ``method='forward'`` remains available for gradient-to-Hessian
cache reuse and validation. Adjoint gradients clear ``energy_hessian`` and
store timing and orbital residual diagnostics in ``derivative_info``; they
do not build the first-response cache required by the forward Hessian.
Hessians also default to ``method='adjoint'``, the analytic Z-vector method
described below. Explicit ``method='polarization'`` and ``method='mixed'``
retain forward response. The default requires optional PyTorch and the
adjoint's supported reference model; unsupported cases raise an error rather
than silently switching methods.

``benchmarks/bse_adjoint_gradients.py`` compares both methods on identical
full-BSE/QP-screened CD/STO-3G references for H3+ and water, using all Cartesian
coordinates and one/two selected states. Reference preparation is excluded.
One warmup precedes three median-timed repetitions per method; each evaluation
starts with an empty forward response cache. The script writes JSON accuracy
and timing results and reproducible PNG/PDF figures. Run with ``PYTHONPATH=.``
and single-thread BLAS/OpenMP; integral construction uses ``pyqed.num_workers()``.

For the larger Cartesian benchmark, run::

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/pyrazine_adjoint_gradients.py \
      --output /private/tmp/pyrazine-ccpvdz-adjoint

This uses the existing idealized planar pyrazine geometry, cc-pVDZ, CD threshold
``1e-10``, exact spectral G0W0 and full BSE with HF-gap screening. Each method
runs in an isolated process and computes the first excitation's total-energy
gradient over all 30 Cartesian coordinates. ``state_ids=[0]`` means the first
excited state, not the RHF ground state. The benchmark verifies identical
reference arrays before comparing gradients. Gradient timings exclude reference
preparation. It reports sampled baseline, peak and incremental RSS (20 ms
sampling), plus the OS process-lifetime peak separately. These are single cold
calls rather than repeated timing averages. JSON progress records and final
PNG/PDF comparisons are written under the requested output directory.

The 2026-10-23 local run resolved the central worker default to **1**, with
one BLAS thread. For the first singlet root and all 30 Cartesian directions,
forward response took 1376.782 s (sampled peak RSS 2.289 GiB), while the adjoint
took 355.071 s (4.122 GiB): a 3.877-fold speedup at higher resident memory.
Incremental peaks above the post-reference baselines were 1.122 and 2.980 GiB,
respectively. Maximum and RMS gradient differences were 2.225e-8 and 9.301e-9
Hartree/bohr. Saved geometry and reference-energy arrays matched exactly.
These are cold single-call timings and agreement with forward response, not
an independent finite-difference accuracy certification. The adjoint used the
shared spectral self-energy pullback, avoiding separate dense inverses per
orbital pair; all 14 adjoint regression tests passed after that change.

Adjoint memory storage
~~~~~~~~~~~~~~~~~~~~~~

The default ``gradient(storage='disk')`` stores large tensors saved for
reverse differentiation in temporary files outside the repository. Values
remain float64, with no approximation, truncation or change in response
equations. Pullbacks map saved tensors as needed; files are removed on both
success and failure. ``storage='memory'`` provides the identical calculation
with saved tensors retained in RAM. The graph is released after the last
selected state. Disk storage trades I/O for lower resident memory; the
existing memory estimate remains conservative and is not a strict RSS cap.

On the same pyrazine/cc-pVDZ first-root Cartesian benchmark, disk-backed saved
adjoints reduced sampled peak RSS from 4.122 to 2.754 GiB (33.2%). Incremental
RSS above the reference baseline fell from 2.980 to 1.602 GiB. The largest
change from the previous adjoint gradient was 1.78e-14 Hartree/bohr; the maximum
forward-response difference remained 2.225e-8 Hartree/bohr. Reference geometry
and energy arrays matched. ``benchmarks/plot_bse_adjoint_memory.py BEFORE AFTER``
reproduces the memory/accuracy comparison from the saved benchmark directories.


Analytic Z-vector Hessians
~~~~~~~~~~~~~~~~~~~~~~~~~~

``excited.hessian(state_ids=[0], modes=modes)`` defaults to
``method='adjoint'`` and evaluates
relaxed surface Hessians and stores both ``energy_gradient`` and
``energy_hessian``. ``modes=None`` selects all Cartesian coordinates;
a mode matrix selects its projected Hessian, including every mixed entry.
The returned shape is ``(nstates, nmodes, nmodes)`` and Cartesian units are
Hartree/bohr². State zero is the first excitation.

This is an analytic forward-over-reverse extension of the gradient adaptation
of Tölle, Kitsaras and Loos cited above, not a reproduction of a published
GW--BSE Hessian algorithm. It differentiates each Z-vector pullback along
one displacement, using the existing first RHF/CPHF response and
``OrbitalResponse`` transpose solves. For an orbital equation
:math:`A^T z=b`, its directional response satisfies
:math:`A^T z_y=b_y-A_y^T z`. Positive matrix-square-root responses and their
pullbacks use Sylvester equations. Quasiparticle response includes the
implicit root derivative and the derivative of its frequency residue.
The BSE eigenvector derivative includes the complete excitation spectrum,
including states not requested by the caller.

For integral variables :math:`I=(h,S,V)` and relaxed gradient coefficients
:math:`\bar I`, each entry is assembled as
:math:`E_{xy}=\bar I_y\cdot I_x+\bar I\cdot I_{xy}+E^{\mathrm{nuc}}_{xy}`.
The AO/factor first derivatives and mixed curvature reuse the existing
analytic response helpers. Each symmetric integral pair is evaluated once;
only its state-contracted scalar is retained for the transposed entry.
CD curvature columns use the existing bounded batching helper. The CD path
pulls the factor covector back through :math:`B=L^{-1}V` once per state,
then contracts packed raw integral curvature directly. First-factor and
Cholesky-root cross terms are retained, but no full second-factor tensor or
per-pair second triangular solve is needed. The generic factor pullback is
shared with the existing RI curvature implementation. The orbital transpose
factorization is reused across all Hessian columns. Screening contractions
are associated in auxiliary space, and quasiparticle directional pullbacks
use bounded orbital blocks with the complete pole space. Denominators are reconstructed per block. Geometry-local pole vectors are
stored in disk-backed blocks and reused, without retaining full orbital-pair/
pole arrays in RAM. Matrix-square-root, quasiparticle and excitation
eigendecompositions are also reused across columns through the existing
reference store. The cache is local to one Hessian call and contains only
zeroth-order values; every directional response is evaluated independently. The quasiparticle
JVP's fixed-root directional vectors are written in blocks to the same
temporary store. The pullback reuses them and applies the quasiparticle-root
shift correction, retaining the full frequency-residue derivative. It does
not repeat the large pole-space products. The quasiparticle context therefore
does not retain the redundant full coupling tensor in Hessian calculations.
Saved backward tensors use copy-on-write mappings; geometry and directional
caches use bounded copies from disk. Immutable NumPy reference inputs are
shared during the call. After large stages and before backward graph nodes,
the existing numerical-allocator release helper returns unused arenas to
the operating system when supported (macOS and glibc Linux); it does not
release live arrays. Backward hooks are removed even if differentiation fails.
Blocked factor contractions use matrix-product association with 64-factor
blocks; quasiparticle contractions use eight-orbital blocks. Neither changes
the complete auxiliary or pole spaces. The factorized RHF potential also has
blocked analytic JVPs and pullbacks, including both occurrences of the factor
and the density response; this avoids full factor-sized Coulomb/exchange
work arrays during automatic differentiation. The two-sided AO-to-MO factor transformation uses
blocked analytic JVPs and pullbacks, retaining both coefficient-response
terms. This extends the existing transformation algebra to forward-over-
reverse differentiation without full auxiliary-factor work arrays. Finished
screening and BSE intermediates are released at their last use.
Completed directional autograd graphs are released
at each column boundary, including reference cycles in temporary Function
classes. These are exact
algebraic rearrangements in float64; they add no rank or response truncation.
No second-order RHF, screening, quasiparticle,
or BSE matrix response is propagated. No nuclear finite differences,
response truncation, or precision reduction enter this path. The independently
computed columns are returned without symmetrization; ``derivative_info``
records their maximum asymmetry and timings by stage.

The gradient's real RHF, exact spectral G0W0, full orbital-space,
nondegenerate-HF-orbital, and fixed-rank CD / unscreened Cholesky RI restrictions
remain. TDA/full BSE and HF/QP screening are supported. Individual surfaces
at excitation degeneracies are NaN. The method still uses dense electronic
matrices and requires first RHF response for every requested direction;
one state-specific gradient solve does not supply an entire Hessian.
Temporary first responses are disk-backed, and saved reverse tensors default
to ``storage='disk'``. ``storage='memory'`` keeps the latter in RAM.
Temporary stores are cleaned on success and failure, and any pre-existing
forward cache is restored. The memory estimate is conservative, not an RSS cap.

Reproduce isolated-process timing, RSS and accuracy comparisons with::

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/pyrazine_adjoint_gradients.py \
      --quantity hessian --system water --output /private/tmp/water-adjoint-hessian

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/pyrazine_adjoint_gradients.py \
      --quantity hessian --directions 3 --output /private/tmp/pyrazine-adjoint-hessian

The water/STO-3G comparison uses all nine Cartesian coordinates. The pyrazine/
cc-pVDZ comparison uses the first nitrogen's x, y, and z displacements and the
resulting complete 3-by-3 Hessian block. Both use CD integrals, full BSE,
HF-gap screening and the first singlet excitation. Each method starts in a
fresh process; reference preparation is excluded from timing. The forward
reference uses ``method='mixed'`` when the memory gate selects bounded
response. The script saves the exact directions, matching reference arrays,
Hessians, timing and sampled/process peak RSS, plus reproducible PNG/PDF figures.
These single cold calls measure method agreement, not repeated-run timing
statistics. Small-system regression tests additionally compare independently
displaced gradients, full Cartesian Hessians and translational invariance.

The initial 2026-10-23 local Hessian benchmark resolved the central worker default to
one worker and used one BLAS thread. For the complete water/STO-3G Cartesian
Hessian, forward response took 0.516 s with sampled peak RSS 0.395 GiB;
the adjoint took 0.839 s and 0.447 GiB. The maximum Hessian difference was
2.414e-8 Hartree/bohr². This small case favors forward response.

For the selected pyrazine/cc-pVDZ 3-by-3 block, forward response took
570.586 s with sampled peak RSS 3.144 GiB; the adjoint took 381.163 s and
5.398 GiB. The adjoint was 1.497 times faster but used more peak memory.
Incremental RSS above the reference baselines was 1.993 versus 4.245 GiB.
Maximum and RMS Hessian differences were 2.657e-7 and 8.872e-8 Hartree/bohr²;
the independently evaluated adjoint columns differed from their transpose
by at most 6.089e-13 Hartree/bohr². Geometry, RHF, MO, QP and excitation
reference arrays matched exactly. The adjoint timings comprised 69.788 s
for first response, 213.773 s for directional reverse response, and
97.582 s for mixed integral curvature and final contractions. Batched CD
curvature improves runtime at a resident-memory cost. These measurements
cover the selected block, not the full pyrazine Hessian; no full-Hessian
speedup or independent pyrazine finite-difference certification is implied.
All 99 focused adjoint, forward and subspace-response tests passed, including
forced disk storage, failure cleanup and displaced-gradient validation.

After the exact contraction and storage optimizations described above, the
same pyrazine block took 317.652 s with sampled peak RSS 4.227 GiB and
incremental peak 3.104 GiB: 16.66% less wall time (1.200 times faster) and
21.71% less peak RSS than the previous adjoint. It is 1.796 times faster
than the saved forward reference, but still uses more peak RSS than forward
(3.144 GiB). The unchanged forward calculation was reused; the optimized
adjoint ran in a fresh process with one worker and one BLAS thread.
Reference arrays matched exactly. Maximum change from the previous adjoint
was 2.179e-11 Hartree/bohr². Maximum/RMS differences from forward were
2.658e-7 / 8.872e-8 Hartree/bohr², and maximum asymmetry was 2.728e-13.
Stage timings were 69.265 s first response, 177.717 s reverse response, and
67.242 s integral contraction. Custom square-root and quasiparticle JVPs
avoid recording unnecessary reverse graphs: their differentiated pullbacks
already supply the required analytic mixed response.

The optimized full water/STO-3G Hessian retained a maximum forward difference
of 2.414e-8 Hartree/bohr²; its 0.989 s / 0.447 GiB adjoint still costs more
than forward (0.549 s / 0.393 GiB). Thus the optimization does not imply a
speed advantage for small systems. The 125 focused tests passed, including
factor-pullback duality and blocking invariance; all 33 adjoint tests were
rerun after the final JVP graph change. The measurements are single calls,
not repeated-run statistics or full pyrazine-Hessian measurements.

Optimized outputs and PNG/PDF figures are in
``/private/tmp/pyrazine-adjoint-hessian-optimized`` and
``/private/tmp/water-adjoint-hessian-optimized``. Reproduce the before/after
figure and machine-readable comparison using::

    MPLCONFIGDIR=/private/tmp/matplotlib-codex PYTHONPATH=. \
      python benchmarks/plot_bse_adjoint_memory.py \
      /private/tmp/pyrazine-adjoint-hessian \
      /private/tmp/pyrazine-adjoint-hessian-optimized

Hessians now default to ``method='adjoint'``. The subsequent memory-focused
revision was measured on the same three pyrazine directions, again with one
worker and one BLAS thread. Sampled peak RSS fell from 4.227 to 3.846 GiB
(9.01%); incremental peak fell from 3.104 to 2.722 GiB. Wall time increased
from 317.652 to 390.987 s, so this is a measured memory/time tradeoff, not
an additional speedup. The optimized call used 328.697 CPU seconds. These
are single-run timings; the saved forward reference remains 570.586 s and
3.144 GiB. No full-pyrazine-Hessian result is implied.

Reference arrays matched exactly. The maximum change from the previous
adjoint was 2.681e-12 Hartree/bohr²; maximum/RMS forward differences were
2.657e-7 / 8.872e-8 Hartree/bohr². Maximum asymmetry was 4.103e-13.
The final 35 adjoint tests passed, including default dispatch, displaced-
gradient Hessian validation, disk/memory agreement, blocking invariance and
release of completed direction data with automatic cyclic collection disabled.
The broader 102-test forward/adjoint/subspace run passed after changing the
API default; forward-reference tests explicitly select their algorithm.

Results and figures are in ``/private/tmp/pyrazine-adjoint-hessian-default``.
Reproduce the comparison with ``benchmarks/plot_bse_adjoint_memory.py`` using
``/private/tmp/pyrazine-adjoint-hessian-optimized`` as BEFORE and this directory
as AFTER. The same Hessian benchmark command above reconstructs the reference
and reruns the selected directions. Explicit ``method='mixed'`` or
``method='polarization'`` remains available for forward response.


A subsequent exact reuse and buffer-lifetime revision recovered the speed
without increasing peak memory on the same selected pyrazine block. The
single-call wall time fell from 390.987 to 208.110 s (1.879x), and sampled
peak RSS fell from 3.846 to 3.582 GiB (6.87%). Incremental peak RSS fell
from 2.722 to 2.473 GiB; the optimized call used 199.263 CPU seconds.
Compared with the earlier 317.652 s / 4.227 GiB revision, this is also
faster and smaller. The saved forward run remains 570.586 s / 3.144 GiB,
so the revised adjoint was 2.742x faster while still using more memory than
forward response. These are single runs on this macOS host, not repeated
statistics or a full 30-coordinate pyrazine Hessian.

The calculation retained the same cc-pVDZ basis, CD rank 1848, HF screening,
first excited state, three Cartesian N1 directions, one worker and one BLAS
thread. Reference arrays matched exactly. The maximum change from the
previous low-memory adjoint Hessian was 4.419e-13 Hartree/bohr²; maximum/RMS
forward differences remained 2.658e-7 / 8.872e-8 Hartree/bohr². Maximum
asymmetry was 6.042e-13 Hartree/bohr². Precision, response terms, integral
thresholds and pole spaces were unchanged.

The implementation reuses geometry-only spectral data and directional QP
blocks, blocks the RHF potential and AO-to-MO pullbacks, avoids redundant
saved couplings, and releases completed first-response arrays after each
direction. Platform allocator relief complements these lifetime reductions;
its RSS benefit is platform dependent. The 38 focused adjoint tests cover
forward comparisons, displaced-gradient validation, disk/memory agreement,
blocking invariance, geometry-cache scope and copy-on-write integrity.

Outputs and PNG/PDF figures are in
``/private/tmp/pyrazine-adjoint-hessian-reuse``. The forward files in that
directory are the saved reference run, not a new forward calculation.
Reproduce the calculation and the before/after figure with::

    MPLCONFIGDIR=/private/tmp/matplotlib-codex PYTHONPATH=. \
      python benchmarks/pyrazine_adjoint_gradients.py --quantity hessian \
      --directions 3 --method adjoint \
      --output /private/tmp/pyrazine-adjoint-hessian-reuse
    MPLCONFIGDIR=/private/tmp/matplotlib-codex PYTHONPATH=. \
      python benchmarks/plot_bse_adjoint_memory.py \
      /private/tmp/pyrazine-adjoint-hessian-default \
      /private/tmp/pyrazine-adjoint-hessian-reuse

Small-system full-Hessian finite-difference validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On 2026-10-24, all nine Cartesian columns of the distorted-water Hessian
were checked against central differences of analytic adjoint gradients at
steps 1e-3, 5e-4, and 2e-4 bohr. Both basis sets used the pyrazine benchmark
settings: CD tolerance 1e-10, no integral screening, RHF tolerance 1e-11,
exact-frequency GW with zero broadening, HF screening energies, and the
low-rank BSE solver with tolerance 1e-9. State index 0 denotes the first
excitation; the differentiated surface is RHF plus that excitation energy.

STO-3G passed: maximum errors over all 81 Hessian entries were 2.6612e-5,
6.5591e-6, and 9.7423e-7 Hartree/bohr², respectively, consistent with
second-order finite-difference convergence. The full analytic Hessian took
4.335 s with sampled peak RSS 0.427 GiB (7 AOs, CD rank 28).

cc-pVDZ failed the same 1e-5 Hartree/bohr² absolute criterion. Maximum
errors were 6.0545e-4, 9.9268e-4, and 2.3498e-3 Hartree/bohr²: smaller
steps made the disagreement worse. Its full Hessian took 16.046 s with
sampled peak RSS 0.468 GiB (24 AOs, CD rank 273). Nevertheless, the full
forward-response and adjoint Hessians agreed to 1.5284e-8 Hartree/bohr²,
and their reference gradients agreed to 1.0358e-8 Hartree/bohr. The forward
Hessian took 10.898 s. Thus agreement between these response formulations
alone does not establish finite-difference accuracy. The source of the
cc-pVDZ discrepancy was unresolved at this stage; the diagnosis and fix
below use this much cheaper investigation case instead of pyrazine.

Timings are single derivative calls, excluding reference construction
(0.118 s for STO-3G and 0.287 s for cc-pVDZ) and interpreter startup. Each
Hessian ran in an isolated process with one worker and one BLAS thread.
Displaced references and CD factors were rebuilt at every point; gradients
reused a process to avoid import overhead, so their individual RSS samples
are not independent memory benchmarks. No response algorithm was changed
for these checks.

Reproduce for each BASIS (``sto-3g`` or ``cc-pvdz``), choosing a fresh OUT::

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/pyrazine_adjoint_gradients.py --system water \
      --basis BASIS --quantity hessian --method adjoint --output OUT
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/validate_pyrazine_hessian.py --output OUT \
      --columns 0 1 2 3 4 5 6 7 8 --steps 0.001 0.0005 0.0002 --reuse-process

For the independent forward comparison, rerun the first command with
``--method forward`` and then run the benchmark with ``--summarize``.
Outputs are in ``/private/tmp/water-hessian-check-sto3g`` and
``/private/tmp/water-hessian-check-ccpvdz``. Reproduce the combined PNG/PDF
figure and JSON comparison with::

    MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/plot_water_hessian_validation.py \
      /private/tmp/water-hessian-check-sto3g \
      /private/tmp/water-hessian-check-ccpvdz \
      --output /private/tmp/water-hessian-check-comparison

CD triangular-factor consistency correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The water/cc-pVDZ discrepancy was traced to ``FactorCurvature`` recovering
its Cholesky root from the stored pivot block. Floating-point elimination
leaves small entries above the theoretically lower-triangular block:
1.13e-11 in this water reference, whose root condition number is about
6.94e5. ``solve_triangular`` ignored those entries, but ordinary matrix
products in the differentiated root recurrence included them. This
inconsistent representation amplified elimination roundoff into integral
response errors. The block first-derivative Coulomb response differed from
the direct differentiated recurrence by 8.26e-8 at the reference geometry;
using the lower triangle consistently reduced that difference to 4.44e-16.

``FactorCurvature`` now explicitly takes the lower triangle of the CD pivot
block. This is an exact structural correction to the existing fixed-pivot
Cholesky response formulation, not a tighter integral threshold, auxiliary
truncation, change of physical model, or finite-difference implementation.
Stored integral factors and reference energies remain unchanged. Tight SCF
convergence, tighter GW roots, and fixed pivot sequences alone did not cure
the original failure. A dense BSE reference also agreed with the iterative
reference, while the error could be reproduced directly in the CD Coulomb
response without GW/BSE.

Repeating all nine water/cc-pVDZ columns with the original settings and
independently rebuilt adaptive CD factors gave maximum errors of 5.9821e-5,
1.6118e-5, and 2.6100e-6 Hartree/bohr² at steps 1e-3, 5e-4, and 2e-4 bohr.
The smallest-step error is about 900 times smaller than before and passes
the unchanged 1e-5 criterion. The errors now decrease approximately
quadratically. RHF energies, orbital energies, QP roots and excitation
energies matched the saved original reference bitwise. The analytic Hessian
changed by at most 2.5330e-6 Hartree/bohr². Its asymmetry is 3.9250e-9 and
translational residual 1.6360e-11 Hartree/bohr².

This single corrected full-Hessian call took 13.926 s with sampled peak RSS
0.489 GiB. These measurements are not repeated-run speed or memory claims.
Outputs are in ``/private/tmp/water-hessian-triangular-fixed``. Use the water
commands above with this fresh output directory and default tolerances.
The cc-pVDZ regression in ``tests/test_bse_adjoint.py`` compares both diagonal
and mixed Hessian entries against displaced analytic gradients.

Reproduce the before/after figure with::

    MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/plot_water_hessian_validation.py \
      /private/tmp/water-hessian-check-ccpvdz \
      /private/tmp/water-hessian-triangular-fixed --labels Before After \
      --output /private/tmp/water-hessian-fix-comparison

The old full pyrazine Hessian has not been rerun with this correction and
must not be treated as validated by the water result.

After the correction, a fresh forward-response run matched the adjoint
Hessian to 1.5284e-8 Hartree/bohr² (RMS 4.9111e-9), with identical reference
arrays. The 63 tests in ``tests/test_bse_adjoint.py`` and
``tests/test_rhf_factor_hessian.py`` passed, including the new displaced-
gradient cc-pVDZ regression and the existing CD/RI curvature checks.

Selected analytic Hessian columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``bse.hessian(method='adjoint', columns=[0, 1, 2])`` evaluates every row of
only the requested columns, returning ``(nstates, nmodes, nselected)``.
Indices refer to ``modes`` when supplied, or to Cartesian coordinates
otherwise. They must be unique and in range; the requested order is retained.
All row-direction first responses are retained, while reverse differentiation
and mixed integral curvature are evaluated only for the selected columns.
The gradient retains all rows. The reported asymmetry applies to the selected
square subblock, not to uncomputed columns. This extends the existing analytic
response implementation without changing its physical approximations.

The benchmark supports ``--columns`` for adjoint Hessians. For selected
pyrazine/STO-3G columns N1-x, N1-y, and N1-z, including all 30 rows::

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/pyrazine_adjoint_gradients.py --system pyrazine \
      --basis sto-3g --quantity hessian --method adjoint --columns 0 1 2 \
      --scf-tol 1e-13 --scf-gradient-tol 1e-12 --qp-tol 1e-11 \
      --output /private/tmp/pyrazine-sto3g-selected-hessian-tight
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python benchmarks/validate_pyrazine_hessian.py \
      --output /private/tmp/pyrazine-sto3g-selected-hessian-tight \
      --columns 0 1 2 --steps 0.001 0.0005 0.0002 --reuse-process

On 2026-10-24, this 30-by-3 pyrazine/STO-3G section passed the displaced-
gradient check after tightening reference convergence. At 2e-4 bohr, the
maximum errors over all 30 rows were 2.8603e-7 (N1-x), 4.7026e-7 (N1-y),
and 2.4991e-7 (N1-z) Hartree/bohr², all below the unchanged 1e-5 criterion.
The x errors at 1e-3 and 5e-4 bohr were 1.2325e-5 and 2.9180e-6;
the y errors were 5.4298e-6 and 1.3539e-6. The z errors were already
3.5021e-7 and 1.2025e-7, so its smallest-step result shows a numerical
floor rather than further quadratic improvement.

The passing reference uses SCF energy tolerance 1e-13, explicit orbital-
gradient tolerance 1e-12, and GW root refinement to 1e-11. BSE tolerance
remains 1e-9; CD tolerance remains 1e-10 with zero screening and adaptive
pivots rebuilt independently at every geometry. There are 34 AOs, 21 occupied
orbitals and CD rank 423. The minimum HF orbital gap is about 4.79e-5 Ha.
The initial run with the original benchmark convergence settings failed its
x-column check (6.24e-5, 6.35e-5, and 1.34e-3 at the three decreasing steps);
its remaining checks were stopped. The tighter analytic section differs
from that reference by at most 2.582e-7 Hartree/bohr². These controls establish
a passing combined convergence protocol, not attribution to a single solver
tolerance. No physical response terms or basis functions were removed.

The selected analytic section took 63.403 s, excluding reference preparation,
with sampled peak RSS 0.587 GiB, one worker and one BLAS thread. Its selected
3-by-3 block asymmetry was 4.137e-13 and its translational residual across
all 30 rows was 6.094e-12 Hartree/bohr². These are single-run measurements.
The 18 displaced full gradients reused one process to avoid repeated imports;
their individual memory samples are not isolated-process benchmarks.
Results, validation JSON, and the reproducible PNG/PDF figure
``pyrazine_selected_hessian_validation`` are stored in
``/private/tmp/pyrazine-sto3g-selected-hessian-tight``. The full Hessian and
cc-pVDZ validation were not rerun as part of this selected STO-3G check.
All 40 tests in ``tests/test_bse_adjoint.py`` passed after adding column
selection, including a comparison of nonconsecutive, reordered selected
columns and their full gradient against the complete directional Hessian.

Selected-column response optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The adjoint Hessian now computes orbital response and MO-transformed electronic
response only for requested columns. All row directions retain their raw
one-electron and factor/integral derivatives, including mixed curvature. This
changes work scheduling only; no response terms or tolerances are dropped.
Full Hessians still require every orbital tangent and do not benefit from this
particular optimization.

On the same tight-tolerance pyrazine/STO-3G 30-row, three-column validation
above, wall time decreased from 63.403 to 51.382 s (1.234 times faster).
First-response preparation decreased from 28.544 to 18.126 s. The Hessian was
bitwise identical, and reusing the saved displaced analytic gradients retained
a maximum finest-step error of 4.703e-7 Hartree/bohr squared. These are individual
runs, excluding reference setup. Peak process RSS was 0.587 versus 0.553 GiB;
process baselines differed, so this is not evidence of a guaranteed reduction
in incremental working memory (123 versus 137 MiB).

Results are in ``/private/tmp/pyrazine-sto3g-hessian-selected-speed``. The
finite-difference point directories there link to the preceding validated run.
``benchmarks/plot_bse_adjoint_memory.py BEFORE AFTER`` reproduces the timing,
peak-memory and numerical-difference figure, with forward comparison included
when the reference directory contains a forward result.

Contracted mixed curvature for full Hessians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CD Z-vector Hessian now passes the packed factor pullback weights to the
existing compiled derivative-column kernel. Second derivatives are accumulated
as scalar direction pairs in one shell pass per state, instead of emitting
batches of rank-by-AO-pair tensors and contracting them afterwards. This is a
late shell contraction of the same unscreened Gaussian derivatives, not a new
integral approximation. First-factor/root cross terms remain explicit. The
triangular-factor consistency fix is unchanged. One-electron curvature reuses
``contracted_one_electron_curvature`` and the existing scalar pair cache.
RI and dense two-electron response retain their existing paths.

These contractions adapt the Gaussian recurrence of Obara and Saika,
J. Chem. Phys. **84**, 3963 (1986), doi:10.1063/1.450106. They change contraction
and storage order only: no response terms, rank thresholds, or derivative
screening are altered. The compiled integral extension must be rebuilt.

A complete 30-by-30 pyrazine/STO-3G Hessian benchmark, using the tight controls
above, state 0 (first excited total-energy surface), one resolved worker and
one BLAS thread, gave:

.. list-table:: Full Hessian before and after contracted curvature
   :header-rows: 1

   * - Metric
     - Previous
     - Contracted
   * - Wall time, s
     - 153.867
     - 100.101
   * - CPU time, s
     - 135.422
     - 62.044
   * - Integral-curvature wall time, s
     - 92.427
     - 14.717
   * - Peak process RSS, GiB
     - 0.591
     - 0.552
   * - Incremental sampled peak, MiB
     - 230.047
     - 81.250

This is a measured 1.54-fold overall wall-time speedup and 6.28-fold curvature
speedup, with 6.6 percent lower peak RSS. These are single isolated-process
runs; reference setup is excluded and process baselines differ. Unchanged
first/reverse response wall times also varied, so the wall-time ratio should
not be interpreted as a universal scaling guarantee. The benchmark does not
establish performance for a full cc-pVDZ Hessian.

All 900 entries agree with the previous implementation to 5.83e-13
Hartree/bohr squared. The maximum asymmetry is 4.70e-13 and translational
residual 6.13e-12. Columns 0, 1 and 2 were compared with the previously saved
analytic-gradient finite differences at 0.001, 0.0005 and 0.0002 bohr; the
finest-step maximum error remains 4.70e-7 Hartree/bohr squared. The displaced
gradients were reused, not rerun. The focused suite passed 55 tests, including
multiple scalar weights, selected/reordered mixed pairs, one/four-worker
kernel checks, forward Hessian references and water/cc-pVDZ finite differences.

Reproduce the optimized full benchmark with::

    PYTHONPATH=. python benchmarks/pyrazine_adjoint_gradients.py \
        --system pyrazine --basis sto-3g --quantity hessian --method adjoint \
        --scf-tol 1e-13 --scf-gradient-tol 1e-12 --qp-tol 1e-11 \
        --output /private/tmp/pyrazine-full-curvature-after

The before/after output directories are
``/private/tmp/pyrazine-full-curvature-before`` and
``/private/tmp/pyrazine-full-curvature-after``. The latter contains comparison
and validation JSON, Hessian arrays, and PNG/PDF diagnostics. Reproduce the
comparison with ``benchmarks/plot_bse_adjoint_memory.py BEFORE AFTER`` and the
finite-difference diagnostics with ``benchmarks/validate_pyrazine_hessian.py``.

Three full cc-pVDZ columns: stage benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contracted-curvature implementation was benchmarked on pyrazine/cc-pVDZ
with all 30 Cartesian rows and columns 0, 1, 2 (N1-x, N1-y, N1-z). This is a
30-by-3 Hessian, not the earlier 3-by-3 directional block. The reference has
104 spatial orbitals, 21 occupied orbitals and 1848 CD factors. The resolved
central worker budget was 1, with one BLAS thread. Controls were CD tolerance
1e-10 with zero integral screening, RHF tolerance 1e-13, RHF gradient tolerance
1e-12, standard GW root tolerance 1e-9 and BSE tolerance 1e-9. State 0 is the
first excited total-energy surface.

An initial attempt to refine all GW roots to 1e-11 failed dominant-root
certification for fragmented core orbital 0 before the Hessian started. The
completed benchmark uses the successful standard 1e-9 GW solve, matching the
previous cc-pVDZ root tolerance. It is not a demonstration of 1e-11 GW accuracy.

.. list-table:: Three full cc-pVDZ columns, isolated process
   :header-rows: 1

   * - Stage
     - Wall time (s)
     - CPU time (s)
     - Peak process RSS (GiB)
   * - First response
     - 479.8
     - 422.6
     - 1.254
   * - Adjoint response
     - 213.4
     - 144.4
     - 1.586
   * - Integral curvature
     - 394.6
     - 245.6
     - 1.231

Total derivative wall time was 1092.047 s (18.201 min), CPU time 812.621 s,
and sampled/OS peak RSS 1.586 GiB. Reference setup took another 27.800 s and
is excluded. Stage wall times come from the solver; CPU spans and RSS samples
are attributed at stage-log boundaries, with 20 ms memory sampling. About
4.3 s of setup/finalization is outside the reported stage wall totals. CPU
spans include boundary overhead. RSS includes shared reference data; the
stage peaks must not be added together.

First response accounts for 43.9 percent of total wall time, curvature 36.1
percent, and adjoint response 19.5 percent. First-response preparation is the
largest time/CPU target; adjoint response sets the memory peak. First-order
integral/factor construction and storage should be profiled next. The current
stage measurements do not isolate individual kernels or attribute wall/CPU
gaps uniquely to disk I/O.

Adjoint column times were 79.2, 68.0 and 66.2 s. Curvature passes took 326.4,
34.6 and 33.6 s; the first includes shared preparation for all requested pairs.
Do not extrapolate a 30-column runtime by multiplying this total by ten.
All 90 entries were finite; selected-block asymmetry was 8.673e-13 and the
translational residual 6.062e-12 Hartree/bohr squared. No new finite-difference
comparison was performed in this timing run.

Reproduce with::

    PYTHONPATH=. python benchmarks/pyrazine_adjoint_gradients.py \
        --system pyrazine --basis cc-pvdz --quantity hessian --method adjoint \
        --columns 0 1 2 --scf-tol 1e-13 --scf-gradient-tol 1e-12 \
        --qp-tol 1e-9 \
        --output /private/tmp/pyrazine-ccpvdz-three-columns-stages-standard-qp

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
        python benchmarks/plot_bse_hessian_stages.py \
        /private/tmp/pyrazine-ccpvdz-three-columns-stages-standard-qp

That directory contains the arrays, reference data, stage measurements and
transition events in ``adjoint.json``, consistency checks and column timings
in ``stage_diagnostics.json``, and ``pyrazine_hessian_stages.png``/``.pdf``.

Packed first-response preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CD first-response path solves the differentiated factor equation in the
existing packed AO-pair representation and expands only its final derivative
for electronic contractions. It uses the same triangular pivot factor and
root derivative as the full-pair solve; this changes data flow, not the
factorization or response equations. RI and dense-integral paths keep their
existing formulations.

The selected-column adjoint Hessian streams up to two first-order integral
directions per shell traversal through the existing disk-backed column helper.
The batch size respects the response-memory allowance and resolved worker
count; a small allowance disables batching. With one worker, first-order
projected integrals accumulate directly into their zero-initialized output,
without a private reduction buffer. Two output directions then occupy the
same integral-output storage as one direction plus its former private buffer.
Multi-worker reductions and second-order kernels are unchanged. Cache ownership
and adjoint tensor storage retain their original behavior.
No screening, CD threshold, GW root selection or response tolerance is relaxed.

To keep the complete Hessian's memory bounded, screening-coupling tangent and
pullback products also use orbital blocks. This reuses the adjoint's existing
custom forward/reverse contraction pattern and the exact factorized coupling
formula used by ``screening.FactorizedCouplings`` (Bruneval, J. Chem. Phys.
136, 194107, 2012, doi:10.1063/1.4718428). That NumPy storage class cannot carry
PyTorch dual derivatives directly, so the same products are expressed as a
custom autograd operation. The full output coupling and all poles are retained;
only product-rule temporaries are blocked. This is an exact reassociation, not
a pole-space approximation. Standalone gradients retain their existing path.

Matched three-column benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fresh isolated-process control used the saved original preparation code and
original integral binary. Both runs used the same 30 rows, columns 0/1/2,
geometry, CD pivots, reference energies, thresholds and one-worker/one-BLAS-thread
settings described above. The final implementation was checked against all 90
control Hessian entries. Its numerical reference arrays and pivots are identical.

.. list-table:: First-response optimization, pyrazine/cc-pVDZ
   :header-rows: 1

   * - Stage
     - Control wall (s)
     - Optimized wall (s)
     - Control CPU (s)
     - Optimized CPU (s)
     - Control sampled RSS (GiB)
     - Optimized sampled RSS (GiB)
   * - First response
     - 361.828
     - 234.247
     - 360.105
     - 232.738
     - 2.742
     - 2.579
   * - Adjoint response
     - 107.501
     - 101.013
     - 94.204
     - 89.817
     - 3.131
     - 3.246
   * - Integral curvature
     - 249.778
     - 244.596
     - 185.669
     - 191.655
     - 1.674
     - 1.611

Total wall time fell from 722.446 to 583.223 s (1.239 times faster), with
first-response preparation 1.545 times faster. Total CPU time fell from
639.977 to 514.208 s. Reference setup is excluded. The first-response sampled
peak fell by 5.95 percent. Integral curvature is now the largest stage at
41.94 percent of total wall time, narrowly ahead of preparation at 40.16 percent.

Overall OS peak RSS was 3.231995 GiB for the control and 3.246124 GiB for the
optimized run, a measured increase of 0.437 percent (14.47 MiB). Thus overall
memory was nearly flat, but this benchmark does **not** demonstrate strict
peak-memory nonincrease. The sampled overall peaks were 3.131 and 3.246 GiB;
20 ms sampling missed a short control peak, so the comparison plot can use
the OS high-water mark explicitly. Stage peaks include shared data and must
not be summed. Historical 1.586 GiB/1092.047 s measurements did not reproduce
in this session and are not used to calculate the optimization's speedup.

Maximum Hessian change was 1.049e-12 Hartree/bohr squared (RMS 2.645e-13),
and maximum gradient change was 2.309e-14 Hartree/bohr. Selected-block asymmetry
was 6.528e-13 and the translational residual 6.269e-12 Hartree/bohr squared.
All 80 focused adjoint, RHF-factor-Hessian and CD-column tests passed, including
forward-response comparisons, blocked-pullback invariance and existing water
finite-difference regressions. This timing run does not add a new pyrazine
finite-difference validation. No precision or convergence tolerance was reduced.
Broader immutable-view/cache-sharing experiments were rejected after higher
RSS measurements; those changes are not retained.

The control and final artifacts are saved under
``/private/tmp/pyrazine-ccpvdz-three-columns-first-preparation-control`` and
``/private/tmp/pyrazine-ccpvdz-three-columns-packed-blocked-coupling``.
They include arrays, timings, source hashes and source snapshots. The control's
``run_control.py`` loads its saved pre-change Python modules and integral binary.
The final directory includes ``preparation_validation.json``,
``memory_comparison.json``, ``stage_diagnostics.json`` and PNG/PDF diagnostics.
Reproduce the final calculation and plots with::

    PYTHONPATH=. python benchmarks/pyrazine_adjoint_gradients.py \
        --system pyrazine --basis cc-pvdz --quantity hessian --method adjoint \
        --columns 0 1 2 --scf-tol 1e-13 --scf-gradient-tol 1e-12 \
        --qp-tol 1e-9 \
        --output /private/tmp/pyrazine-ccpvdz-three-columns-packed-blocked-coupling

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
        python benchmarks/plot_bse_adjoint_memory.py \
        /private/tmp/pyrazine-ccpvdz-three-columns-first-preparation-control \
        /private/tmp/pyrazine-ccpvdz-three-columns-packed-blocked-coupling \
        --memory-metric process_peak_rss_bytes

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
        python benchmarks/plot_bse_hessian_stages.py \
        /private/tmp/pyrazine-ccpvdz-three-columns-packed-blocked-coupling

Projected iterative BSE eigenvector response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BSE.hessian(excitation_response='projected')`` replaces the complete
excitation-eigenvector response sum with a projected linear solve. The default
``'dense'`` remains the reference implementation. This option applies to the
adjoint Hessian; it does not change the GW model, screening poles, CD rank,
or excitation space.

For full BSE, the implementation reuses ``_bse_full_matvec`` from the existing
factorized BSE eigensolvers and SciPy GMRES, the same iterative backend used
by ``qchem.orbital_response``. For right state :math:`z=(X,Y)` and normalized
left state :math:`l=\eta z/(z^T\eta z)`, with
:math:`\eta=\mathrm{diag}(I,-I)` and :math:`Q=I-zl^T`, it solves

.. math::

   [Q(L-\omega I)Q+zl^T]\,\partial z=-Q(\partial L)z,
   \qquad \partial\omega=l^T(\partial L)z.

The projector and shifted operator are actions, not assembled matrices.
Both positive- and negative-frequency sectors remain in the solve. TDA uses
the existing TDA action with the Euclidean projector. The physical full-BSE
tangent is transformed to the existing symmetric representation using
:math:`S=\sqrt{A-B}` and its already computed derivative:

.. math::

   \partial u=S^{-1}[\sqrt{\omega}(\partial X+\partial Y)
      -(\partial S)u]+u\,\partial\omega/(2\omega).

This is an exact differentiated-eigenproblem adaptation of the adjoint
formulation documented above, not a truncated sum over states. GMRES follows
Saad and Schultz, SIAM J. Sci. Stat. Comput. 7, 856--869 (1986),
doi:10.1137/0907058, using diagonal preconditioning and a restart space capped
at 80 vectors. ``response_tol`` defaults to 1e-11; ``response_maxiter`` defaults
to 200 restart cycles. True projected and physical response residuals, metric
orthogonality, reference residuals and iteration counts are recorded in
``derivative_info['excitation_response_diagnostics']`` by state and column.
Unconverged or failed residual checks raise; there is no silent dense fallback.

This is **not a fully matrix-free Hessian**. GW spectral screening, A/B
first derivatives and matrix-square-root differentiation remain dense. Full
BSE reuses the complete spectrum already required by the square-root operation
for the existing 1e-8 degeneracy test, avoiding another eigenvector basis and
without weakening the spectral-isolation check. TDA retains a dense spectral isolation and state-refinement check. Individual surfaces at degeneracies retain the existing
undefined/NaN behavior. No new screening-pole or low-energy-state approximation
is introduced.

Regression checks cover dense/CD/RI integrals, HF/QP screening, full BSE/TDA,
multiple requested states, reordered states and selected columns, a near-gap
model, failed-solver rejection, and the water/cc-pVDZ finite-difference test.

Matched pyrazine/cc-pVDZ benchmark (2026-10-25)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both modes computed all 30 rows of columns 0, 1 and 2 (first N atom xyz),
state 0 (first excited total-energy surface), with 104 spatial orbitals,
21 occupied orbitals and 1848 CD factors. Runs used the central one-worker
setting and one BLAS thread, identical source hashes, reference arrays and
CD pivots. Reference construction is excluded from derivative wall time;
OS peak RSS covers the process.

.. list-table:: Single matched pair
   :header-rows: 1

   * - Metric
     - Dense
     - Projected
   * - Wall time (s)
     - 605.82
     - 602.77
   * - CPU time (s)
     - 523.80
     - 529.90
   * - OS peak RSS (GiB)
     - 3.293
     - 3.278
   * - First response (s)
     - 243.48
     - 243.27
   * - Reverse response, three columns (s)
     - 105.63
     - 113.58
   * - Mixed integral curvature (s)
     - 253.02
     - 242.19

The maximum absolute difference across all 90 Hessian entries is
2.3672e-11 Hartree/bohr squared (RMS 4.8377e-12). GMRES took 21, 18 and 24
iterations, 7.70 seconds combined, with true relative projected residuals
below 1.81e-12. The projected water/cc-pVDZ finite-difference regression also
passes. This benchmark compares against the dense analytic Hessian, not a
new pyrazine finite-difference calculation.

The 0.5 percent total wall-time and peak-RSS differences do not establish a
performance improvement from one pair. The reverse-response stage is slower
with projected solves; variation in the unchanged curvature stage offsets it.
Dense remains the default. Removing the extra excitation eigenvector basis
alone does not remove the dense GW and square-root bottlenecks.

Artifacts are in ``/private/tmp/pyrazine-ccpvdz-dense-response`` and
``/private/tmp/pyrazine-ccpvdz-projected-response``; the latter includes
``projected_response_validation.json``, ``memory_comparison.json`` and
PNG/PDF figures. Reproduce each run by substituting dense or projected::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
    python benchmarks/pyrazine_adjoint_gradients.py \
        --system pyrazine --basis cc-pvdz --quantity hessian --method adjoint \
        --excitation-response projected --columns 0 1 2 \
        --scf-tol 1e-13 --scf-gradient-tol 1e-12 --qp-tol 1e-9 \
        --output /private/tmp/pyrazine-ccpvdz-projected-response

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
    python benchmarks/plot_bse_adjoint_memory.py \
        /private/tmp/pyrazine-ccpvdz-dense-response \
        /private/tmp/pyrazine-ccpvdz-projected-response \
        --memory-metric process_peak_rss_bytes --labels Dense Projected

Physical XY Hessian contractions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``excitation_response='physical'`` reuses projected GMRES but differentiates

.. math::

   \Omega = X^T A X + Y^T A Y + 2X^T B Y,
   \qquad X^T X-Y^T Y=1,

with frozen reverse-mode amplitudes and their analytic forward tangents.
Stationarity in the BSE metric makes this equivalent to the symmetric
square-root representation for an isolated real root. It removes both BSE
matrix-square-root JVPs and pullbacks. A geometry-cached, nondifferentiated
full-spectrum calculation retains the positive-matrix and degeneracy checks.
TDA uses the corresponding normalized X quadratic form.

The static screening contraction uses the existing ``StaticScreening``
dielectric identity (Bruneval, JCP 136, 194107, 2012,
doi:10.1063/1.4718428), adapted to the adjoint tensor graph. With orbital-gap
matrix D, transition Coulomb matrix C and bare transition couplings E,

.. math::

   M W^{-1}M^T = E^T(D+4C)^{-1}E.

For CD/RI, the contraction is associated through the auxiliary metric
:math:`V_{ov}(D+4V_{ov}^T V_{ov})^{-1}V_{ov}^T`, avoiding an additional
all-orbital-pair static coupling tensor. The implementation uses the existing
inverse differentiation path: the direct linear-solve primitive gave correct
values and JVPs but incorrect mixed pullbacks in the supported Torch path.
Dense/CD/RI HF/QP-screening Hessian comparisons cover this distinction.
This is an exact algebraic adaptation, with no pole truncation, precision
reduction, modified CD threshold or omitted response terms. Dynamic GW still
retains the complete dense pole representation and its root derivatives;
this option does not make the entire GW-BSE Hessian matrix-free.

All adjoint response variants now refine their derivative states using the
geometry-local spectrum already needed for isolation checks. The physical
path reuses projected GMRES for residual corrections (up to four checks,
requiring a residual below 1e-13) instead of constructing a full Casida
eigenvector basis for refinement. The input BSE
energies/amplitudes remain unchanged. The dense sum and the projected tangent
use these stationary vectors and matching eigenvalues consistently. Only
selected XY vectors are retained by the physical path, not a second full
excitation basis. Diagnostics record ``stationary_reference=True`` and the
iterative response records the small ``reference_energy_shift``.

This refinement matters for pyrazine/cc-pVDZ: with the original BSE residual
of about 2e-10, the direct XY and symmetric-root Hessians differed by 1.39e-7
Hartree/bohr squared despite gradients agreeing to 6.3e-13 Hartree/bohr.
An isolated-entry diagnostic using a stationary reference (residual about
2.3e-13) reduced that difference to 3.1e-12. The initial physical benchmark
under ``/private/tmp/pyrazine-ccpvdz-physical-response`` is therefore marked
rejected; it is not the final accuracy validation. Tests explicitly perturb
input amplitudes and verify that refinement restores the Hessian without
mutating the supplied BSE reference.

For CD/RI, the final static contraction evaluates the equivalent auxiliary
form :math:`R=[I-(I+4V_{ov}D^{-1}V_{ov}^T)^{-1}]/4`. Its custom analytic
JVP and pullback reuse the coupling code's auxiliary blocking pattern.
The pullback accumulates one full factor-gradient array, avoiding padded
full-factor copies from separate occupied/virtual slice adjoints. Orbital-gap
and factor responses, including their mixed derivatives, are retained.
Tests force small auxiliary blocks and compare both gradients and Hessians
against the dense spectral formulation.

Final blocked physical-response benchmark (2026-10-25)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fresh stationary-state dense control and the final blocked physical path
computed the same full 30-by-3 pyrazine/cc-pVDZ block, columns 0, 1 and 2,
state 0 (first excited total-energy surface). Both used 104 orbitals,
21 occupied orbitals, 1848 CD factors, the central one-worker setting and
one BLAS thread. Reference arrays, CD pivots, settings and integral binary
match. Source snapshots verify that the final change is confined to the
physical static-screening pullback; the control's dense path is unchanged.

.. list-table:: Final comparison; reference setup excluded from timing
   :header-rows: 1

   * - Metric
     - Dense response
     - Physical XY, blocked pullback
   * - Total wall time (s)
     - 565.67
     - 547.71
   * - Total CPU time (s)
     - 499.37
     - 492.49
   * - OS peak RSS (GiB)
     - 3.212
     - 3.190
   * - First response wall time (s)
     - 229.58
     - 227.29
   * - Reverse response wall time (s)
     - 98.57
     - 90.97
   * - Reverse response CPU time (s)
     - 85.09
     - 82.47
   * - Integral curvature wall time (s)
     - 234.40
     - 226.34

The maximum difference over all 90 Hessian entries is 7.9116e-12
Hartree/bohr squared (RMS 2.0477e-12); the maximum gradient difference is
1.1191e-13 Hartree/bohr. The selected-block asymmetry is 8.04e-13 and the
translation residual is 8.30e-12 Hartree/bohr squared. Projected responses
use 21, 18 and 24 iterations, with relative residuals below 2.29e-12;
the refined reference residual is 3.04e-15. The refined energies differ
from the input reference by only 3.94e-15 Hartree.

This single pair measures 7.7 percent less reverse-response wall time and
3.2 percent less total wall time, with essentially unchanged OS peak RSS
(0.7 percent lower). Reverse CPU time decreases by 3.1 percent. Some total
wall-time improvement occurs in unchanged integral stages and must not be
attributed to the electronic optimization. The integral stages still consume
about 83 percent of total time. Dynamic GW remains dense and complete;
there is no screening-pole approximation. The physical option remains opt-in.

An intermediate unblocked static pullback passed accuracy but increased peak
RSS to 3.651 GiB; its validation record is marked rejected. The final blocked
pullback eliminates that increase. Validation includes the 109-test focused
suite, three added perturbed-reference checks, and 17 affected cases rerun
for the blocked pullback, including dense/CD/RI, HF/QP screening, BSE/TDA,
ordinary gradients, Hessians, block boundaries and water finite differences.
The large-system accuracy comparison is against the analytic dense Hessian;
no new pyrazine finite-difference calculation is claimed.

Control artifacts are in
``/private/tmp/pyrazine-ccpvdz-bounded-dense-response``; final artifacts are in
``/private/tmp/pyrazine-ccpvdz-blocked-physical-response``. The latter includes
``physical_response_validation.json``, ``memory_comparison.json``, source
snapshots and PNG/PDF timing, memory and consistency figures. Reproduce with::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
    python benchmarks/pyrazine_adjoint_gradients.py \
        --system pyrazine --basis cc-pvdz --quantity hessian --method adjoint \
        --excitation-response physical --columns 0 1 2 \
        --scf-tol 1e-13 --scf-gradient-tol 1e-12 --qp-tol 1e-9 \
        --output /private/tmp/pyrazine-ccpvdz-blocked-physical-response

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
    python benchmarks/plot_bse_adjoint_memory.py \
        /private/tmp/pyrazine-ccpvdz-bounded-dense-response \
        /private/tmp/pyrazine-ccpvdz-blocked-physical-response \
        --memory-metric process_peak_rss_bytes --labels Dense Physical

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
    python benchmarks/plot_bse_hessian_stages.py \
        /private/tmp/pyrazine-ccpvdz-blocked-physical-response

For a new control run, change ``--excitation-response physical`` to
``--excitation-response dense`` and use a separate output directory.

Selected-column CD curvature contractions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CD factor curvature now packs the AO-pair covector before multiplying by
Cholesky-root derivatives. Packing is linear and acts on AO indices, so it
commutes with multiplication on the auxiliary index. Full Hessians therefore
use ``nao*(nao+1)/2`` columns instead of ``nao**2`` for these products.

For a small selected-column set, the code also moves the products to that
side using the exact Frobenius identity

.. math::

   \langle-L_x^T U,B_y\rangle
     =\langle L_x,-U B_y^T\rangle.

Writing W for the metric pullback, the complete factor-curvature correction
is evaluated as

.. math::

   C_{xy}=\langle L_x,-U B_y^T-2W^T L_y\rangle
           +\langle B_x,-L_y^T U\rangle.

Only the requested columns' two coefficient matrices are prepared and saved
when twice their count is smaller than the row count. Otherwise the full
packed-row formulation is used, avoiding duplicate full-Hessian work.
This is an exact contraction-order change to the existing fixed-pivot CD
curvature; it changes no screening threshold, factor rank, response term or
precision. Row integral and orbital-response preparation is unchanged.
``derivative_info['curvature_timings']`` separates factor-weight preparation,
contracted CD integral evaluation and one-electron curvature. Remaining time
in the integral-curvature stage includes scalar contractions and nuclear terms.

Tests compare both strategies against the original full, unpacked formula,
including multiple states, reordered columns and nonsymmetric test covectors.
Selected-column molecular tests retain all rows and compare against full
small-molecule Hessians; existing water finite-difference checks remain active.

A pyrazine/cc-pVDZ benchmark of all 30 rows and columns 0, 1, 2
(state 0, physical excitation response, 104 orbitals, 1848 CD factors,
one central-budget worker and one BLAS thread) gave:

.. list-table:: Packed selected-column curvature benchmark
   :header-rows: 1

   * - Measurement
     - Previous
     - Optimized
   * - Hessian wall time (s; reference setup excluded)
     - 547.713
     - 515.151
   * - CPU time (s)
     - 492.487
     - 453.293
   * - First response (s)
     - 227.291
     - 233.069
   * - Reverse response (s)
     - 90.974
     - 105.223
   * - Integral curvature (s)
     - 226.342
     - 173.004
   * - OS process peak RSS (GiB)
     - 3.190
     - 2.771

The curvature stage is 23.6% faster; total wall time falls 5.9%. Unchanged
first/reverse stages varied between runs. The overall memory peaks occur in
reverse response, so their decrease cannot be attributed entirely to this
curvature change. Curvature-stage sampled absolute peaks were 1.817 and
1.849 GiB; peaks above stage entry were 334 and 230 MiB, respectively.
The optimization introduces no larger coefficient buffers and stores fewer
selected-column weights. These single-run measurements are not a statistical
performance guarantee or a full 30-column speedup estimate.

All 90 Hessian entries agree within 1.78e-15 Hartree/bohr^2 (RMS 4.34e-16);
gradients are bitwise identical. Reference arrays agree within 1e-10, CD
pivots are identical, and calculation settings and compiled integral kernel
hashes match. The focused adjoint/factor-Hessian/CD-column suite passed
112 tests, followed by five targeted tests covering the updated selected-row
case and both packed contraction strategies.

First-response preparation is now the largest stage (233.1 s). Within
curvature, contracted integral evaluation takes 94.3 s, factor weights 37.0 s,
and one-electron curvature 1.5 s; scalar contractions and other overhead
account for the remaining 40.2 s. Dynamic GW response remains dense.

Reproduce with the physical-response benchmark command above, changing
``--output`` to ``/private/tmp/pyrazine-ccpvdz-curvature-columns``. Compare
that directory against ``/private/tmp/pyrazine-ccpvdz-blocked-physical-response``
using ``benchmarks/plot_bse_adjoint_memory.py`` and generate stage figures
with ``benchmarks/plot_bse_hessian_stages.py``. The new output directory
contains source snapshots/hashes, arrays, ``curvature_columns_validation.json``,
its companion ``validate.py``, and PNG/PDF diagnostic figures.

Shared triangular first-factor response
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First CD factor derivatives now reuse the connection formulation already
used by ``CoulombIntegrals``. For a fixed auxiliary space,

.. math::

   B=L^{-1}V,\qquad M=LL^T,\qquad
   \Phi_x=\operatorname{lower}_{1/2}(L^{-1}M_xL^{-T}),

   L_x=L\Phi_x,\qquad B_x=L^{-1}V_x-\Phi_xB.

Here ``lower`` retains the lower triangle and halves its diagonal. This
avoids computing ``L_x B`` and subsequently solving through ``L`` for that
product. The shared helper in ``eri_response`` uses triangular BLAS for
``Phi_x B``; the GW/BSE response also uses it for ``L Phi_x``. Packed AO
pairs are retained until the existing electronic contractions require the
expanded factor. Inputs are never modified, and no factor rank, precision,
screening or response threshold changes. This is an exact algebraic
reorganization of the fixed-auxiliary CD formulation of Aquilante, Lindh
and Pedersen cited above, with its same fixed-pivot restrictions.

``derivative_info['first_response_timings']`` separates batched integral
column generation, unselected row-factor preparation, and selected electronic
response. The difference from total first-response time includes one-electron
row integrals, remaining storage and memory-release overhead.

On the same full 30-row, three-column pyrazine/cc-pVDZ benchmark (columns
0, 1, 2; physical excitation response; state 0; one worker and one BLAS
thread), this change reduced first-response wall time from 233.069 to
217.920 s (6.5%) and first-response CPU time from 232.582 to 216.123 s.
Its sampled stage peak RSS fell from 2.513 to 2.419 GiB. Overall Hessian
wall time was 515.151 versus 487.183 s (5.4% shorter), with total CPU time
453.293 versus 428.111 s and OS peak RSS 2.771 versus 2.767 GiB.
Unchanged reverse-response and curvature stages also varied between runs;
the total speedup should not be attributed solely to this algebra change.
Reference setup is excluded, and these are single-run measurements.

Preparation now comprises 131.263 s of integral-column generation,
41.403 s of unselected row-factor preparation, 30.491 s of selected
response, and 14.763 s of remaining overhead. Integral-column generation
is therefore the largest remaining preparation cost (60.2%).
All 90 Hessian entries agree with the previous implementation within
2.284e-12 Hartree/bohr^2 (RMS 4.478e-13); gradient differences are at most
2.487e-14 Hartree/bohr. Reference settings, arrays and CD pivots match.
Selected-block asymmetry is 8.25e-13 and the translation residual is
5.46e-12 Hartree/bohr^2. The focused adjoint, factor-Hessian and CD-column
suite passed 114 tests. Six subsequent targeted checks include independent
displaced Cholesky factorizations, empty rank, and selected-column molecular
Hessians. Full 30-column pyrazine performance was not measured.

Reproduce using the same benchmark command with output directory
``/private/tmp/pyrazine-ccpvdz-triangular-first`` and compare it with
``/private/tmp/pyrazine-ccpvdz-curvature-columns``. The output contains
``first_preparation_validation.json``, ``validate.py``, source snapshots and
hashes, arrays, and the PNG/PDF comparison and stage figures generated by
``benchmarks/plot_bse_adjoint_memory.py`` and
``benchmarks/plot_bse_hessian_stages.py``.

Dependency-selected first integral columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The compiled first-derivative CD column path now reuses the dependency planner
already used for second derivatives. It collects shifted HRR expansions needed
by the selected shell targets, evaluates only their recurrence ancestors, and
shares each expansion value across derivative recipes within a primitive
quartet. Previously this path filled the complete rectangular recurrence table
and repeatedly evaluated common expansions. This is an exact scheduling
adaptation of the Gaussian recurrence of Obara and Saika, J. Chem. Phys.
84, 3963 (1986), doi:10.1063/1.450106; no requested term is dropped.
The fixed-CD-pivot restrictions remain those of the factor response above.

The existing dependency-plan cache remains limited to 256 plans or 8 MiB
per worker. First-derivative direction batches and integral output allocation
are unchanged; this path adds no primitive-batch tensor. Rys and scalar/dense
reference paths, screening thresholds, factor ranks and response tolerances
are unchanged. Uncontracted first-column ``derivative_info`` now reports
nonzero plan-hit/miss counts, like the second-derivative path.

The two-direction pyrazine/cc-pVDZ kernel probe took 10.193 s with the saved
previous extension and 6.320 s with the new extension. Both visited 374073
quartets, screened none, and returned exactly equal integral values. The new
path recorded 137726 dependency-plan hits and 3488 misses.

The full 30-row, three-column Hessian benchmark (columns 0, 1, 2, state 0,
physical excitation response, 1848 CD factors, one worker/BLAS thread) gave:

.. list-table:: First-integral dependency scheduling benchmark
   :header-rows: 1

   * - Measurement
     - Previous
     - Optimized
   * - First integral-column generation (s)
     - 131.263
     - 82.333
   * - First-response preparation (s)
     - 217.920
     - 168.632
   * - Reverse response (s)
     - 101.871
     - 97.449
   * - Integral curvature (s)
     - 163.822
     - 164.000
   * - Total Hessian wall time (s)
     - 487.183
     - 433.594
   * - Total CPU time (s)
     - 428.111
     - 377.519
   * - Preparation sampled peak RSS (GiB)
     - 2.419
     - 2.418
   * - Whole-process sampled peak RSS (GiB)
     - 2.767
     - 2.784
   * - OS lifetime peak RSS (GiB)
     - 2.767
     - 2.863

Integral generation is 37.3% faster, preparation 22.6% faster, and the complete
three-column calculation 11.0% faster. All 90 Hessian entries and all gradient
entries are bitwise identical to the previous run. Reference arrays, CD pivots,
and numerical settings match; extension hashes intentionally differ and are
recorded. The focused CD-column, RHF factor-Hessian and BSE adjoint suite passed
116 tests, including Cartesian/spherical and worker comparisons, scalar
integral references and water finite differences.

Memory limits and batch sizes were unchanged. Preparation's sampled peak
was unchanged, but the OS lifetime peak increased by 98.84 MiB (3.5%).
The largest sampled peak was in reverse response; 20-ms samples do not locate
the higher OS transient. These single-run measurements therefore do not
establish unchanged overall memory usage. First-response preparation and
integral curvature now have similar costs. Full 30-column Hessian timing
was not measured, and reference setup is excluded from the wall timings.

Results and source snapshots are in
``/private/tmp/pyrazine-ccpvdz-first-recurrence``. Repeat the benchmark command
above using that output directory, and compare against
``/private/tmp/pyrazine-ccpvdz-triangular-first`` with
``benchmarks/plot_bse_adjoint_memory.py``; stage diagnostics use
``benchmarks/plot_bse_hessian_stages.py``. The output includes
``integral_generation_validation.json``, ``validate.py``, the kernel probe
and its script, and reproducible PNG/PDF comparison and stage figures.

Shared first-integral output across workers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uncontracted first CD columns now accumulate into one common output array
instead of a complete array per worker followed by reduction. The existing
AO-pair projection is reused; 256 column-lock stripes protect overlapping
Cartesian-to-spherical contributions. A worker holds only one stripe at a
time, releasing it before the next column. Worker-local recurrence/HRR caches
and small scalar reductions remain unchanged. This changes accumulation
ownership and addition order only: no screening, precision reduction or
omitted derivative terms. Multi-worker results can differ at floating-point
roundoff level because their addition order is not fixed.

``CoulombIntegrals.derivative_info['private_output_bytes']`` reports the
aggregate private output allocation. It is zero for uncontracted first
columns, independent of worker count; it excludes worker-local recurrence
workspaces. First-direction batching now budgets one output copy, with the
existing two-direction cap. Second-derivative batching retains its original
private-reduction accounting; contracted curvature has scalar-sized output.
For two pyrazine/cc-pVDZ directions, rank 1848 and 5460 AO pairs, removing
four private outputs eliminates 616 MiB of integral-output storage.

An isolated pyrazine/cc-pVDZ two-direction benchmark, using one BLAS thread,
measured:

.. list-table:: First-integral worker output memory
   :header-rows: 1

   * - Configuration
     - Integral time (s)
     - OS peak RSS through kernel (GiB)
     - Sampled RSS above entry (GiB)
   * - One worker, shared output
     - 6.602
     - 0.701
     - 0.280
   * - Four workers, previous private outputs
     - 2.297
     - 1.422
     - 0.998
   * - Four workers, shared output
     - 2.404
     - 0.834
     - 0.412

Four-worker peak RSS falls 41.3%, with 4.6% more kernel time in this single
comparison. The shared four-worker kernel remains 2.75 times faster than the
serial kernel. All saved integral arrays are bitwise identical (including
the previous four-worker output), and pivots match. This does not guarantee
bitwise reproducibility for arbitrary parallel sums. First-column private
output storage is exactly zero according to the new allocation diagnostic.
The focused 120-test suite passed, followed by seven targeted checks of dense
displacement directions and water/cc-pVDZ Hessians; the physical-response
water Hessian agrees between one and four integral workers within 1e-10.

These measurements cover two Cartesian first-integral directions, not the
complete 22-mode Hessian. Timing excludes molecule construction, output saving
and numerical comparison. RSS is sampled at 5 ms; the OS peak is read after
the kernel and before saving/comparing arrays. Worker-local workspaces still
increase memory relative to one worker. No full-Hessian memory estimate should
be inferred directly from this isolated kernel measurement.

Reproduce each case in a separate process using
``benchmarks/benchmark_cd_worker_memory.py --workers N --output DIR``.
``--extension PATH`` selects a saved control binary and ``--reference DIR``
validates against previously saved columns after measurement. The saved
control is in ``/private/tmp/bse-first-memory-before`` and the optimized
binary/source snapshots in ``/private/tmp/bse-first-memory-after``.
Run ``benchmarks/plot_cd_worker_memory.py SERIAL_DIR PREVIOUS_DIR SHARED_DIR
--output PLOT_DIR`` to regenerate the PNG/PDF comparison. This run used
``/private/tmp/pyrazine-cd-memory-serial``,
``/private/tmp/pyrazine-cd-memory-private4``, and
``/private/tmp/pyrazine-cd-memory-shared4``; the plots and validation summary
are in ``/private/tmp/pyrazine-cd-worker-memory``.

Exact full-pole GW response in a fixed spectral frame
----------------------------------------------------

Adjoint gradients and Hessians accept ``gw_response='spectral'``. This is
an exact algebraic reorganization of the existing zero-broadening spectral
G0W0 response, retaining every screening pole and the existing QP root and
response tolerances. ``gw_response='dense'`` remains the default and the
independent control. This option can be combined with physical, projected,
or dense excitation response, HF or QP screening, and dense, CD, or RI ERIs.
The approximations of the selected GW/BSE model and integral representation
are unchanged.

In the fixed eigenbasis of the reference positive Casida matrix :math:`C`,
let :math:`\omega_a=\sqrt{\lambda_a(C)}`. For each orbital pair, the
self-energy matrix function is

.. math::

   f(C)=C^{-1/2}(\delta I+s C^{1/2})^{-1},\qquad
   f[\lambda_a,\lambda_b]
   =-\left(s+\frac{\delta}{\omega_a+\omega_b}\right)f_a f_b.

Here :math:`s=+1` for occupied internal orbitals and :math:`s=-1` for
virtual internal orbitals. The first square-root response is obtained from
:math:`\Omega\Omega_x+\Omega_x\Omega=C_x`. Differentiating the transpose
Sylvester equation supplies the mixed response. The fixed-frame rotations
remain inside the differentiation graph, including off-diagonal Casida
responses. Repeated positive screening poles therefore require no
eigenvector derivatives or eigenvalue-gap divisions. Isolated QP roots,
positive screening eigenvalues, and the existing BSE state restrictions
are still required.

This replaces nested square-root/inverse derivative graphs and repeated
orbital-pair pole rotations with divided differences and a Sylvester
pullback. It is **not matrix-free GW**: the complete reference eigenspectrum,
dense Casida response contractions, and all poles remain. No finite-chain
Lanczos or polynomial approximation is introduced. The formulation is an
independent response adaptation of the spectral GW equations discussed by
Tölle, Kitsaras, and Loos, *J. Phys. Chem. Lett.* **16**, 11134 (2025),
DOI: `10.1021/acs.jpclett.5c02219 <https://doi.org/10.1021/acs.jpclett.5c02219>`_,
not a reproduction of that work's Lagrangian implementation.

Validation includes a noncommuting perturbation at repeated screening poles,
finite differences of weighted QP-root gradients, dense/CD/RI Hessian
comparisons with both screening choices, and water/cc-pVDZ finite differences
of analytic gradients. Use ``benchmarks/pyrazine_adjoint_gradients.py
--gw-response spectral`` to select this response in the benchmark driver.

For CD/RI integrals the implementation forms orbital-pair couplings in blocks
and contracts their adjoints directly into factors and projection weights;
it does not retain a full bare coupling and its full pullback simultaneously.
Reference pole vectors and directional vectors use the existing response
storage. This reduces temporary storage relative to the first unfused
spectral implementation, but does not establish a lower peak than the dense
control.

A fresh isolated-process pyrazine/cc-pVDZ comparison (all 30 rows, columns
0, 1, 2, state 0, physical excitation response, 1848 CD factors, one worker
and one BLAS thread) measured:

.. list-table:: Full-pole GW response comparison
   :header-rows: 1

   * - Measurement
     - Dense control
     - Blocked spectral
   * - Total wall time (s)
     - 450.518
     - 447.056
   * - Total CPU time (s)
     - 385.002
     - 376.420
   * - First response wall time (s)
     - 161.549
     - 173.658
   * - Reverse response wall time (s)
     - 99.924
     - 90.979
   * - Integral curvature wall time (s)
     - 185.682
     - 178.542
   * - OS peak RSS (GiB)
     - 2.911
     - 2.970

Reverse-response wall/CPU time fell 9.0%/9.2%, but total wall time fell only
0.8% and OS peak RSS increased 2.0%. The unchanged preparation and curvature
stages varied substantially between runs. These single-run measurements
**do not demonstrate a useful end-to-end speedup or a memory reduction**;
the dense GW response remains the default. The earlier unfused spectral
version took 451.550 s and peaked at 3.081 GiB. Full matrix-free GW response
remains unimplemented by this option.

All 90 Hessian entries agree with the fresh dense control to
1.600e-11 Hartree/bohr squared (RMS 4.570e-12); the maximum gradient difference
is 2.487e-14 Hartree/bohr. Pivots, reference energies, integral binary hashes,
worker counts, and tolerances match. The focused 99-test suite passes,
including water/cc-pVDZ finite-difference validation and repeated-pole tests.
Reference preparation is excluded from timing; OS peak RSS includes it.

Reproduce the two runs with ``benchmarks/pyrazine_adjoint_gradients.py`` and
``--system pyrazine --basis cc-pvdz --quantity hessian --method adjoint
--excitation-response physical --columns 0 1 2 --scf-tol 1e-13
--scf-gradient-tol 1e-12 --qp-tol 1e-9``, selecting ``--gw-response dense``
or ``--gw-response spectral`` and separate output directories. Saved runs
are ``/private/tmp/pyrazine-ccpvdz-dense-gw-control`` and
``/private/tmp/pyrazine-ccpvdz-fused-spectral-gw-response``. The latter contains
``validation.json``, ``memory_comparison.json``, and PNG/PDF diagnostic
figures. Regenerate figures with ``benchmarks/plot_bse_adjoint_memory.py
DENSE_DIR SPECTRAL_DIR --memory-metric process_peak_rss_bytes
--labels Dense Spectral`` and ``benchmarks/plot_bse_hessian_stages.py
SPECTRAL_DIR``. Source snapshots and the validation script are in
``/private/tmp/bse-gw-response-after``.

Matrix-free shifted-solve GW first-response prototype
----------------------------------------------------

``pyqed.gw.shifted_response.ShiftedGW(energy, factors, nocc)`` evaluates the
correlation self-energy and a supplied directional first derivative from
real spatial MO energies and CD/RI MO factors. It reuses
``lanczos_response.casida_action`` and the SciPy restarted GMRES backend
already used by contour and orbital response. Neither a transition-space
Casida matrix nor an auxiliary dielectric matrix is assembled. No dense
screening eigenspectrum, square root, inverse, or static Cholesky
preconditioner is needed. Factors remain resident; solver scratch scales
with the configured restart length, rather than the full transition-space
square. Orbital pairs and quadrature shifts are processed sequentially.

This is an independent resolvent adaptation of the restricted direct-RPA
Casida formulation (Bruneval, *J. Chem. Phys.* **136**, 194107 (2012),
DOI: `10.1063/1.4718428 <https://doi.org/10.1063/1.4718428>`_) with restarted
GMRES (Saad and Schultz, *SIAM J. Sci. Stat. Comput.* **7**, 856 (1986),
DOI: `10.1137/0907058 <https://doi.org/10.1137/0907058>`_). It does not
reproduce either paper's complete GW algorithm or claim their performance
guarantees.

For :math:`C=D^2+4\sqrt{D}F^T F\sqrt{D}`, define
:math:`\delta=\omega-\epsilon_q`, :math:`a=|\delta|`, and :math:`s=+1`
for occupied intermediate orbitals, :math:`s=-1` for virtual ones. For
nonzero :math:`a`, the numerically stable representation is

.. math::

   h(C)b=\frac{2a}{\pi}\int_0^\infty
          \frac{(C+t^2I)^{-1}b}{a^2+t^2}\,dt,
   \qquad
   f_s(C,\delta)b=\begin{cases}
       s h(C)b,&s\delta>0,\\
       s[2(C-a^2I)^{-1}b-h(C)b],&s\delta<0.
   \end{cases}

The correlation self-energy is :math:`2\sum_q b_{pq}^T f_s(C,\delta)b_{pq}`,
with :math:`b_{pq}=\sqrt{D}F^T L_{pq}`. The continuous zero-frequency-difference
limit is :math:`f_s(C,0)b=sC^{-1}b`, with frequency derivative
:math:`-C^{-3/2}b`; the latter uses the positive-shift inverse-square-root
integral. Differentiation of each solve reuses its operator:

.. math::

   (C+t^2I)x'=b'-C'x,\qquad
   (C-a^2I)u'=b'-C'u+2\delta\delta'u.

Quadrature weights, orbital energies, both factor occurrences in the
couplings, and the Casida action are differentiated. The integration mapping
is :math:`t=\alpha\tan\theta`, with a fixed local scale
:math:`\alpha=\sqrt{a\min(D)}` (or :math:`\min(D)` at zero). Holding this
mapping fixed while differentiating is valid for the exact integral;
quadrature error is separately converged. Gauss-Legendre order doubles until
both value and first derivative change by at most ``tolerance``. Nodes use
a banded routine, not a dense screening eigensolve. Every GMRES solve checks
the original equation's relative residual against ``solve_tol``. Failure at
either gate raises; there is no dense fallback or silent root substitution.

This representation avoids artificial opposite-sign poles introduced by
rationalizing :math:`C^{-1/2}(\delta I+s\sqrt C)^{-1}` directly. True
self-energy poles remain ill-conditioned. Residual and order-doubling checks
are not rigorous forward-error bounds; near-pole accuracy still requires
validation. This prototype is not a production QP/root-selection method.
It supports zero broadening only, retains all transitions and intermediate
orbitals, and has no Lanczos-chain or polynomial truncation. Finite
quadrature and iterative solves introduce errors which must be converged.

Example, with already prepared spatial MO factors and their derivative::

   from pyqed.gw.shifted_response import ShiftedGW
   response = ShiftedGW(energy, factors, nocc, solve_tol=1e-12,
                        tolerance=1e-10)
   sigma, dsigma = response.evaluate(p, omega,
       energy_tangent=denergy, factor_tangent=dfactors)
   sigma, frequency_slope = response.evaluate(p, omega,
       frequency_tangent=1.)
   print(response.info)

The supplied derivatives may come from the existing analytic orbital and
integral response infrastructure. The prototype itself does not generate
nuclear/orbital derivatives, static exchange or mean-field corrections, QP
root selection, BSE responses, or Hessians. Existing production defaults are
unchanged. ``benchmarks/benchmark_gw_shifted_response.py --basis BASIS
--output DIR`` builds water and obtains a relaxed H1-y displacement response
from the existing infrastructure, then compares HOMO/LUMO self-energies at
two frequencies each, for nuclear and frequency derivatives, against an
independent full-spectrum analytic divided-difference reference. The dense
matrices in that script are validation controls only. The script writes
``validation.json`` and reproducible PNG/PDF error/timing figures; timing
excludes reference preparation. It does not measure peak memory.

Validation on asymmetric water used one relaxed H1-y displacement (Cartesian
coordinate 4), HOMO and LUMO, and frequencies 0.07 Hartree below and 0.03
Hartree above each orbital energy. Both nuclear and frequency derivatives
were checked, giving eight evaluations per basis. With ``solve_tol=1e-12``
and ``tolerance=1e-10``, the maximum errors were:

.. list-table:: Matrix-free versus dense first-response validation
   :header-rows: 1

   * - Basis
     - Self-energy (Hartree)
     - Nuclear derivative (Hartree/bohr)
     - Frequency derivative (dimensionless)
     - Eight shifted evaluations (s)
     - Eight dense evaluations (s)
   * - STO-3G
     - 4.11e-14
     - 1.21e-13
     - 2.45e-14
     - 3.333
     - 0.00208
   * - cc-pVDZ
     - 2.15e-13
     - 2.69e-13
     - 1.08e-13
     - 40.287
     - 0.02126

All independently checked solve residuals were below 1e-12; quadrature used
64--256 nodes at convergence. These small-system timings show that the
prototype is much slower than the dense reference, not a speed improvement.
No memory benchmark or production QP/Hessian integration is claimed.
The initial direct rationalization failed frequency-derivative convergence
on cc-pVDZ because of cancellation around an artificial pole; the stable
representation above passed with unchanged tolerances.

Saved inputs, numerical reports, source hashes, and figures are in
``/private/tmp/water-sto3g-shifted-gw`` and
``/private/tmp/water-ccpvdz-shifted-gw``. Tests additionally compare the shared
Casida action derivative with finite differences, check the self-energy
against the existing spectral GW routine, test repeated poles, zero
couplings, the continuous zero-delta limit, an artificial-pole location,
and a nearby true pole, and require explicit failure for unconverged
solves/quadrature. A guard test disables dense eigensolvers and inverses
while evaluating the prototype.

Shared shifted solves and scalar first-response adjoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ShiftedGW.evaluate_many(orbital, frequencies, energy_tangent=de,
factor_tangent=df)`` returns one row per requested frequency, with columns
``(self_energy, structural_response, frequency_slope)``. Within each orbital
pair and quadrature order, all frequencies share the positive shifts, using
the smallest nonzero frequency distance to set their common integration
scale. Both responses reuse the same primal solution through the exact
symmetric scalar-adjoint identity

.. math::

   x=A^{-1}b,\qquad (b^TA^{-1}b)'=2b'^Tx-x^TA'x.

This eliminates tangent linear solves entirely, including for the true-pole
correction. Its frequency derivative uses :math:`2\delta x^Tx` for
:math:`A=C-\delta^2 I`. The continuous zero-delta limit is evaluated explicitly.
Only the current solution (plus the zero-shift solution when needed) and
scalar frequency accumulators are retained. There is no shift-by-vector
cache and no persistent cache across calls. Construct a new operator when
reference energies or factors change. Scratch remains bounded by the
existing restarted GMRES workspace, with additional scalar storage linear
in the number of requested frequencies. It does not retain Krylov bases
across solves or interpolate between frequencies.

``evaluate`` now uses this scalar-adjoint path by default, returning the
same requested directional combination. ``evaluate(..., reuse=False)``
keeps the original independent shifted/tangent solves as a validation
control. All values, structural responses, and frequency slopes must meet
the unchanged order-doubling tolerance; every linear residual is still
checked. The additional slope check can require a higher quadrature order
than a structural-only request previously needed. The scalar identity is
an exact algebraic reorganization of the formulation above, not a new
physical approximation; its literature references and near-pole limitations
are unchanged. The production GW/BSE gradient and Hessian paths do not use
this prototype.

``benchmarks/benchmark_gw_shifted_reuse.py`` measures a fresh process for
each method on saved ``response_inputs.npz`` from the preceding validation.
It excludes dense validation from timing and OS-peak measurement. Optional
``--trace-workspace`` separately audits traced allocations and distorts
timing, so it must not be used for speed comparisons. The plotting command
``benchmarks/plot_gw_shifted_reuse.py DIRECTORY`` checks agreement and writes
a numerical comparison and PNG/PDF timing, RSS, and accuracy diagnostics.

On the same saved water/cc-pVDZ inputs, four HOMO/LUMO frequencies with
both the H1-y structural derivative and frequency slope gave the following
fresh-process comparison (one BLAS thread, solve tolerance 1e-12,
quadrature tolerance 1e-10):

.. list-table:: Shared versus independent shifted solves
   :header-rows: 1

   * - Measurement
     - Independent
     - Shared scalar adjoint
   * - Wall time (s)
     - 42.037
     - 11.922
   * - CPU time (s)
     - 42.591
     - 12.046
   * - Shifted solves
     - 92210
     - 17678
   * - GMRES iterations
     - 623213
     - 173992
   * - OS peak RSS (MiB)
     - 72.234
     - 71.375
   * - Sampled RSS above entry (MiB)
     - 4.078
     - 3.734

The shared path is 3.53 times faster here with no measured memory increase.
The small RSS difference is a single-run observation, not a general memory
reduction guarantee. Shared versus dense maximum errors are 4.54e-15 Hartree
for self-energy, 3.97e-15 Hartree/bohr for the structural derivative, and
1.99e-13 for the frequency slope. Differences from independent solves are
below 2.70e-13. The shared batches converged at 256 and 128 quadrature nodes;
no convergence tolerance was loosened. This remains a prototype benchmark,
not a speedup over dense GW or an integrated Hessian calculation.

Results and plots are in ``/private/tmp/gw-shifted-reuse``. Reproduce each
method in a separate process with ``benchmarks/benchmark_gw_shifted_reuse.py
--inputs /private/tmp/water-ccpvdz-shifted-gw/response_inputs.npz
--method METHOD --output /private/tmp/gw-shifted-reuse/METHOD.json``, where
METHOD is ``independent`` or ``shared``, followed by
``benchmarks/plot_gw_shifted_reuse.py /private/tmp/gw-shifted-reuse``.

For an isolated dense-versus-shared benchmark, prepare identical molecular
inputs with ``benchmarks/benchmark_gw_shifted_response.py --system pyrazine
--basis cc-pvdz --coordinate 1 --prepare-only --output DIRECTORY``. Preparation
uses the existing analytic orbital/integral first response; coordinate 1 is
N1-y for the supplied pyrazine geometry. N1-x gives a symmetry-zero scalar
self-energy derivative and is therefore a weaker test of first-response accuracy. It saves both MO factors and their
derivatives, not just orbital energies. The common preparation time is
recorded separately and excluded from response evaluation timing.

The ``--method dense`` option in ``benchmark_gw_shifted_reuse.py`` builds the
complete Casida spectrum and its first-response matrix once, then reuses them
for all HOMO/LUMO frequencies and both derivatives. It does not repeatedly
diagonalize for each request. The shared run accepts ``--reference
DIRECTORY/dense.json`` to validate against that saved result without building
any dense screening matrix in the measured process. Input SHA-256 hashes,
frequencies, dimensions, solver diagnostics, wall/CPU time, process peak RSS,
and sampled RSS are recorded. The RSS monitor retains only a running maximum;
its own storage does not grow with calculation duration. Both response paths
include their operator/spectral setup in timing. Neither includes loading
common input files or the preceding molecular calculation. OS peak RSS is a
process-lifetime measure, including input loading; baseline and sampled peaks
are provided separately.

The pyrazine/cc-pVDZ benchmark used HOMO/LUMO indices 20 and 21, two
frequencies per orbital (HF energy minus 0.07 and plus 0.03 Hartree), and
the nonzero N1-y nuclear response. It retained all 104 intermediate orbitals,
1743 transitions, and 1848 CD factors. Separate processes used one BLAS
thread, identical saved inputs, solve tolerance 1e-12, and quadrature
tolerance 1e-10. The common molecular/first-response preparation took
34.980 s and is excluded below.

.. list-table:: Pyrazine/cc-pVDZ response evaluation on prepared inputs
   :header-rows: 1

   * - Measurement
     - Dense
     - Shared matrix-free
   * - Wall time (s)
     - 2.050
     - 1442.505
   * - CPU time (s)
     - 2.076
     - 1297.654
   * - Process peak RSS (MiB)
     - 688.359
     - 471.000
   * - Sampled RSS above entry (MiB)
     - 314.453
     - 98.859

Both targets converged at 128 quadrature nodes. The shared path performed
49936 shifted solves and 502230 GMRES iterations. Maximum errors versus dense
were 2.134e-16 Hartree for correlation self-energy, 1.301e-15 Hartree/bohr
for its N1-y derivative, and 1.131e-14 for its frequency slope. All original
equation relative residuals were below 1e-12. Eleven focused tests passed,
including the batched dense reference versus the independent spectral formula.

Thus the shared prototype used 31.6% less process peak memory but took
703.8 times longer for this response stage. It is not a speed improvement at
this size. The remaining repeated shifted solves make Hessian integration
premature as a performance optimization. This comparison excludes static
exchange, QP root finding, and Hessian assembly; it does not claim a complete
matrix-free molecular preparation pipeline or end-to-end memory reduction.
No production GW/BSE Hessian defaults were changed.

Saved inputs, raw reports, convergence histories, summary, source hashes,
and PNG/PDF figures are in ``/private/tmp/pyrazine-ccpvdz-shifted-n1y``.
Use ``--method dense --output DIRECTORY/dense.json`` for the control and
``--method shared --reference DIRECTORY/dense.json --output
DIRECTORY/shared.json`` for the matrix-free run, both with ``--inputs
DIRECTORY/response_inputs.npz``. Regenerate the figure with
``benchmarks/plot_gw_shifted_reuse.py DIRECTORY --control dense --title
'Pyrazine/cc-pVDZ: dense vs shared matrix-free GW response'``.

Early contraction of projected CD curvature
-------------------------------------------

Contracted second integral derivatives now accumulate the supplied observable
sensitivities into a 9 by 9 center-derivative block for each shell quartet.
Only this block is projected onto the requested directions. Previously each
shell target was projected separately before applying its sensitivity. For
one or two electronic observables this removes repeated mode-pair work and
replaces shell-target-sized derivative storage with observable-sized storage.
The existing Obara--Saika recurrence (J. Chem. Phys. 84, 3963, 1986,
DOI: 10.1063/1.450106), fixed CD pivots, and unscreened second derivatives are
unchanged: this is an exact algebraic reassociation up to floating-point
summation order, not a new approximation to the reference method.

``CoulombIntegrals._cd_columns(..., order=2, sensitivities=...)`` uses this
path by default; ``early_contraction=False`` retains the late reference.
It supports dense direction vectors, selected pairs, and multiple observables.
No full Cartesian Hessian is needed to form a requested normal-coordinate
block. Dimensionless directions must include the mass and frequency scaling;
the existing pyrazine bath driver supplies this scaling and excludes 6a/10a.
This does not change the definition or geometry of the supplied normal modes.
First-response preparation retains its bounded two-direction batches.

``benchmarks/benchmark_projected_curvature.py`` measures first-column
preparation and selected projected curvature separately, using a reproducible
synthetic sensitivity. Run ``--prepare`` once, then run the default and
``--late`` in separate processes against the same ``--modes`` file.
``benchmarks/plot_projected_curvature.py OUTPUT`` verifies agreement and draws
timing, process-memory, and error figures. These are isolated integral-stage
measurements, not end-to-end GW--BSE Hessian timings.

A one-worker pyrazine/cc-pVDZ benchmark at the unoptimized example geometry
used 22 dimensionless RHF modes (excluding the assigned 6a and 10a modes)
and 63 unique pairs covering the first three columns. Both paths used the
same normalized random sensitivity, fixed pivots, and unscreened integrals.
The late/early curvature wall times were 195.268/126.149 s (1.548x speedup);
CPU times were 144.471/103.879 s (1.391x). Maximum absolute contraction
difference was 1.906e-17. The two first-integral directions were bitwise equal;
their CPU times were 23.942/24.020 s, consistent with unchanged preparation.

Overall process peak RSS was 896.938/787.000 MiB, while sampled curvature-stage
peak RSS was 481.906/561.391 MiB. Thus this run does not establish a uniform
stage-by-stage RSS reduction, despite the smaller shell workspace and lower
overall peak. Twenty-five integral tests and two end-to-end Hessian tests
passed, including water/cc-pVDZ finite differences of analytic gradients.
Raw inputs, reports, and figures are in ``/private/tmp/projected-curvature``.

First-response batching and compact derivative workspaces
--------------------------------------------------------

The first CD integral columns now reuse the primitive-batch evaluator already
used for curvature. All requested HRR contributions are retained; short or
unsupported shell contractions use the existing scalar path. Column outputs
still use the bounded two-direction batching and shared worker output.

First column recurrences reuse the compact dependency layout
already implemented for three-center derivatives: only nodes feeding the
requested HRR terms are stored. The HRR indices are remapped into this layout
while cached geometry plans retain their original indices. This is an exact
storage/scheduling adaptation of the same Obara--Saika recurrence cited above,
with no change to screening, CD pivots, precision, or derivative terms.
``recurrence_workspace_bytes`` reports the sum of worker peak allocations for
scalar/batched recurrence tables, index scratch, and batched HRR outputs. It excludes recipe
caches, integral outputs, and Python/allocator overhead; it is not process RSS.

Curvature-weight preparation packs AO pairs in rank blocks and releases mapped
first-response inputs after packing them. Matrix-product outputs are scaled
in place instead of making negated/scaled copies of the factor inputs. This
avoids full-rank packing intermediates and retaining the last mapped input
into the next stage.
``benchmarks/benchmark_cd_weight_preparation.py`` isolates these contractions
at pyrazine/cc-pVDZ dimensions with synthetic factors; its optional
``--implementation`` path loads an unchanged saved module for comparison.

The final one-worker benchmark against the saved pre-change extension used
the same 22 dimensionless pyrazine/cc-pVDZ modes and 63 selected curvature
pairs. Two first-integral directions took 23.731/14.208 s before/after
(1.670x faster); CPU time was 22.407/13.609 s. Curvature integral wall time
was 102.341/116.017 s and CPU time 98.541/100.613 s. Compact second-column
storage was not retained because its trial increased CPU cost. First-column
maximum error was 3.123e-17 and curvature error 2.033e-19. The 31 focused
integral, packing, and end-to-end Hessian checks passed, including
water/cc-pVDZ finite differences.

In a separate synthetic weight-preparation benchmark with 1848 factors,
104 AOs, three stored directions, and one selected column, stage peak RSS
fell from 801.359 to 481.562 MiB (39.9 percent). Raw weights, row products,
connections, and packed first factors were bitwise equal. CPU time was
3.663/3.585 s; wall time was 3.968/5.207 s. This is a memory improvement,
not evidence of a weight-preparation wall-time speedup.

Memory reductions are not uniform across stages: the integral benchmark's
curvature-stage RSS was 526.297/652.172 MiB and overall process peak was
839.203/849.109 MiB. Thus neither an overall memory reduction nor an
end-to-end Hessian speedup is established by these isolated measurements.
The confirmed improvements are first-integral speed and weight-preparation
memory. Raw arrays, reports, source hashes, and reproducible PNG/PDF figures
are in ``/private/tmp/response-prep``. Use ``--tag before``/``--tag after``
and ``--save-first`` with the integral benchmark, then
``benchmarks/plot_projected_curvature.py /private/tmp/response-prep
--control before --candidate after``. The plotting script also validates
and plots ``weights-before.json``/``weights-after.json`` from the weight
benchmark when present.

Curvature workspace lifetime
----------------------------

The derivative dispatcher retains a fixed scalar recurrence table for second
columns while keeping compact, demand-sized tables for first columns.
Scalar and batched HRR evaluation now share one worker-local buffer, reused
across shell quartets. The batched path no longer also allocates an unused
scalar HRR array. Buffer growth avoids geometric over-allocation beyond the
existing 32 MiB batched-column allowance. These are workspace-lifetime
changes to the same recurrence and projected contractions; pivots, screening,
precision, and derivative terms are unchanged.

Second-column tasks group equal angular classes and nuclear centers before
ordering by primitive cost. Each worker keeps numerical HRR plans only for
its current center group; geometry-independent symbolic and recurrence plans
remain reusable. This bounds the lifetime of geometry-specific cache entries
without changing any integral or discarding response terms.

``benchmarks/benchmark_projected_curvature.py --extension KERNEL`` supports
isolated before/after extension builds. ``--skip-first`` provides a separate
process control for curvature without prior first-response allocation. The
benchmark records curvature-entry RSS as well as stage peak RSS, allowing
retained memory to be distinguished from growth during curvature. Repeated
same-node comparisons and their controls can be plotted using
``benchmarks/plot_curvature_regression.py DIRECTORY``.

The controlled Linux comparison (Westlake xnode102, Slurm job 13276457,
four workers, single-threaded BLAS) used identical builds and inputs in
separate processes. Medians of two runs were:

.. list-table:: Four-worker integral benchmark
   :header-rows: 1

   * - Stage
     - Old baseline
     - Before fix
     - After fix
   * - Two first directions (s)
     - 5.251
     - 3.763
     - 3.710
   * - 63 curvature pairs (s)
     - 18.349
     - 18.444
     - 17.840
   * - Curvature peak RSS (MiB)
     - 665.3
     - 662.8
     - 641.8

Thus curvature time and stage peak RSS fell by 3.3 and 3.2 percent from
the pre-fix implementation, preserving the first-integral improvement.
The maximum curvature difference across these runs was 1.993e-18;
first-integral differences were at most 3.470e-17, with identical pivots.
These are projected integral contractions with synthetic sensitivity,
covering the first three columns of 22 dimensionless modes at the existing
unoptimized pyrazine/cc-pVDZ reference geometry, not full electronic Hessians.
The original large macOS wall-time/RSS regression was not reproduced in
this controlled Linux comparison. Separate curvature-only processes also
showed no substantial first-response memory-retention penalty on Linux.
Buffer reuse alone (job 13276447) did not improve performance; grouping and
limiting the numerical HRR cache provided the measured gain.

The one-worker guard (job 13276459, one run per variant) gave curvature
times of 56.448/56.288/56.850 s and peak RSS of 469.1/473.0/468.6 MiB for
old baseline/pre-fix/fixed. Curvature values were bitwise identical. Thus
serial memory also decreased, but this run showed a 1 percent timing cost,
not a serial speedup. First-integral times were 17.449/11.089/11.046 s.
Plot this guard alongside the repeated comparison with
``--serial /private/tmp/curvature-regression/serial/summary.json``.

All 36 focused local checks passed, including water/cc-pVDZ displaced-gradient
Hessians, projected-vs-Cartesian curvature, one/four-worker execution, and
RHF factor Hessians. Source hashes, reports, comparison drivers, and raw
arrays are retained under the isolated cluster directory
``/share/home/gubingLab/gubing/pyqed_runs/curvature-regression-20261025``;
collected reports and reproducible PNG/PDF figures are under
``/private/tmp/curvature-regression/final``. The shared cluster source and
environment were not modified.

Persistent adjoint workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a complete calculation with automatic workspace cleanup, use::

    gradient, hessian = bse.derivatives(
        order=2, state_ids=[0], modes=modes,
        excitation_response='physical', gw_response='dense')

This evaluates the Hessian once and returns its accompanying gradient; it
avoids a separate gradient evaluation. Existing compatible intermediates
can be reused, then derivative caches and their temporary files are released
on success or failure. The returned arrays remain available on
``energy_gradient`` and ``energy_hessian``. ``order=1`` returns just the gradient
with the same cleanup policy. Method options follow ``gradient``/``hessian``;
``columns`` selects Hessian columns while retaining every gradient direction.
The underlying derivative formulation and restrictions are unchanged.


``BSE.gradient()`` retains a geometry-local workspace that a subsequent
``BSE.hessian()`` can reuse, including for a reordered subset of the supplied
normal modes. It stores reference spectral intermediates, the RHF orbital
solver, state-specific Z-vectors, and raw first integral/factor derivatives.
The gradient does not solve directional CPHF equations or evaluate second
integrals. The Hessian still evaluates its required directional electronic
responses and differentiates the pullbacks. Temporary autograd graphs and
Hessian tangents are freed after each call; a whole reverse graph is not kept
alive between calls.

Both calls must use matching ``gw_response`` and ``excitation_response``
settings. Reference contents, requested states, response tolerances and
contraction block settings enter the cache fingerprint. Large arrays are
stored in a temporary disk workspace. Only one reference/configuration is
retained per BSE object, and accumulating more than a Cartesian basis-sized
set of different directions replaces that workspace. Call
``bse.clear_derivative_cache()`` to release its files explicitly. A failed
adjoint calculation also discards its workspace.

Derivative validation must follow the same quasiparticle branches at displaced
geometries. Independently selecting the largest-weight Dyson root at every
point can switch satellite branches in fragmented virtual orbitals; resulting
finite differences need not approach the derivative of the reference branch.
Branch tracking is a validation control, not a change of the default
largest-weight GW selection policy. See the benchmark's saved root shifts,
residuals and step-size comparisons before interpreting a discrepancy.

A local four-worker pyrazine/cc-pVDZ/CD check (26 October 2026) at the
optimized RHF geometry gave 15.57 s electronic reference preparation,
155.96 s for the 22-mode S1 gradient, and 120.57 s for the subsequent
three-mode Hessian (indices 0, 4, 9). The same Hessian from a fresh derivative
workspace took 142.12 s; caching reduced its wall time by 15.2 percent.
Gradient plus Hessian peaked at 4.10 GiB process RSS. Modes 6a and 10a
(indices 21 and 17) were excluded. These are single-run local timings;
geometry optimization and normal-mode generation are excluded.

The warm and cold matrices agreed within 7.1e-16 Hartree/Q^2. The three
checked gradient components agreed with forward response within 3.4e-14
Hartree/Q, and all three Hessian columns agreed with branch-following
finite differences within 2.1e-8 Hartree/Q^2 at a 0.001 displacement.
Halving the first-column step reduced its error to 3.9e-9. This validates
that three-mode block, not the full 22-mode Hessian. The reproducible driver
is ``benchmarks/pyrazine_bath_adjoint_local.py``; its prepare stage generates
the modes, and its plot stage writes derivative and validation figures.

Package maintenance notes
-------------------------

The unfinished DMFT translation is preserved in
``docs/reference/dmft_unfinished.txt`` as source material, not an available
Python implementation. The experimental SCGW builder explicitly selects the
exact integral path; this does not make its frequency quadrature exact.

On macOS, source builds share PyTorch's bundled OpenMP runtime when PyTorch is
visible to the build environment. This prevents duplicate-runtime aborts when
GW kernels and PyTorch execute in the same process. Rebuild extensions after
changing that runtime or installing PyTorch. An explicit
``PYQED_OPENMP_PREFIX`` overrides automatic selection; its runtime must also be
compatible with other libraries used in the process. For a pip source build alongside PyTorch, install the build dependencies in
the target environment and use ``pip install --no-build-isolation .`` so runtime
discovery sees that environment.
