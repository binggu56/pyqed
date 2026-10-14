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
-----------------------------------

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
-----------------------

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
complete Casida diagonalization. The implementation integrates on a mapped
Gauss–Legendre grid and adds the occupied/virtual poles inside the deformed
contour. Subtracting the zero-frequency screened interaction treats the
orbital-pole step analytically, stabilizing evaluation at mean-field orbital
energies. It uses no analytic continuation, pole truncation, or additional
RI rank approximation.

Screening workspace scales quadratically in the auxiliary count rather than
the occupied-virtual pair count. MO factors still occupy cubic storage, and
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
~~~~~~~~~~~~~~~~~~~~~

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

A separate numerical issue was exposed by the changed floating-point
summation order: the old unconstrained secant solve could select a
negative-weight quasiparticle crossing. Molecular diagonal Dyson roots
now use analytic self-energy derivatives and coupling continuation from
zero to full self-energy, checking positive residue and an absolute
1e-9 Hartree equation residual. Every coupling step uses an outward
bracket search from the previous root, accepting the nearest found upward
crossing in the first successful shell; the coupling step is reduced if no
acceptable bracket is found. Unrestricted Newton steps previously selected
different positive-weight branches under BLAS summation perturbations.
The search uses 120 logarithmic shells and can miss narrow crossings;
it does not enumerate every root or guarantee the globally nearest crossing.
At a branch fold this can select a neighboring positive-weight branch;
it is not a guarantee of uninterrupted analytic continuation. Failure raises
instead of silently replacing the result with a mean-field energy. ``qp_weights`` and ``qp_residuals``
record these checks for the last Dyson solve (before any evGW damping).
This is a numerical branch-selection adaptation of Hedin's Dyson equation, *Phys. Rev.* **139**, A796 (1965),
https://doi.org/10.1103/PhysRev.139.A796; it is not a complete satellite
solver and does not guarantee the root with largest spectral weight.

Spectral GW orbital OpenMP
~~~~~~~~~~~~~~~~~~~~~~~~~

``GW(mf, threads=4).run()`` parallelizes independent diagonal Dyson solves
using the compiled ``dyson`` algorithm. The default is one thread. This
uses the same bracket-first Hedin-equation adaptation as the Python
reference; safeguarded Newton steps stay inside a selected bracket.
It changes numerical branch selection, not the spectral self-energy
formula, and retains every screening pole. The above branch-selection
limitations remain applicable. Contour GW does not support this control.

Factorized orbital couplings are formed in bounded batches, limited by
``max_memory`` and the requested thread count, without materializing the
complete orbital-pair/pole tensor. BLAS is held at one thread during these
transforms; OpenMP orbital workers evaluate scalar pole sums without BLAS.
Casida screening still follows the caller's BLAS settings. ``threading_info``
reports OpenMP availability, requested and observed thread counts, and the
orbital batch size. Memory limits or small batches can reduce observed
parallelism. ``max_memory`` is a batch-storage budget, not a process RSS cap.
At least one orbital's residue array must fit in memory even when its size
exceeds that budget.

The GW extension is included in default extension builds. Set
``PYQED_GW_OPENMP=required`` when building to require OpenMP, or ``0`` to
build its serial compiled path. macOS runtime discovery reuses the existing
build helper and honors ``PYQED_OPENMP_PREFIX``. With the extension absent,
one-thread solves use the Python reference; requesting multiple threads
raises. ``benchmarks/gw_thread_scaling.py`` validates against a profiled
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
-----------------------

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
-------------------------------------

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
--------------------------------------------

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
--------------------------------------------------

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
-------------------------------------------------------

A small nonlinear QP residual establishes convergence to a fitted function,
not accuracy of its analytic continuation. Molecular Padé now checks the
requested fit against neighboring even point counts (normally 16 and 20
for an 18-point fit). A failed comparison, nonpositive-weight comparison
root, or change larger than ``stability_tol=1e-4`` Ha rejects the affected
orbitals. Their energies/weights/residuals become NaN and the solver remains
unconverged. Candidate energies and per-orbital changes remain in
``gw.ac_info['stability']`` for diagnosis. Passing this sensitivity heuristic
is not a real-axis accuracy certificate and does not establish accuracy at
the tighter nonlinear root tolerance.

The default raises instead of silently accepting these roots or launching a
potentially very expensive calculation. To repair rejected states explicitly::

    gw = GW(mf, freq_int='ac', threads=8)
    gw.run(nw=100, on_unstable='contour')

This hybrid retains accepted Padé states and uses the existing complete-RI
contour-deformation solver for rejected targets, including its positive-weight
coupling-continuation branch prescription. Subsequent ``gw.sigma`` evaluations
for repaired states use the contour model too. It adapts the formulation of
Zhu and Chan, JCTC 17, 727–741 (2021), doi:10.1021/acs.jctc.0c00704; it is not
a new exact continuation method. All intermediate orbitals remain in screening
and the self-energy. Repair now uses a contour-specific regulator of
``eta=1e-6`` Ha rather than inheriting the AC object's ``eta=1e-2`` Ha.
It doubles the quadrature and independently reduces eta tenfold, repeating
until both QP changes are below ``accuracy_tol=1e-5`` Ha. After at most
``max_refinements=3`` unsuccessful rounds it raises and leaves rejected
energies unset. The final regulator, grid, and per-orbital changes are stored
in ``gw.ac_info['contour_repair']``. These controls can be supplied directly
to ``repair_continuation``. The check establishes numerical stability under
these refinements, not that the tracked root has the largest spectral weight
or that retained Padé states are exact.
A successful repair can be a weak satellite. Repaired roots with
``qp_weights < 0.01`` now emit a warning and are listed in the repair audit's
``weak_root_orbitals`` field, with ``weak_root_weight_threshold=0.01``.
This is a diagnostic threshold, not a universal quasiparticle criterion:
it neither changes the chosen root nor certifies a dominant peak above the
threshold. ``converged`` continues to describe numerical convergence.
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
It must construct the direct contour screening needed for those targets.

``stability_tol=None`` explicitly disables this safeguard and records the
unchecked status. It is intended for controlled diagnostics and reproduction
of historical Padé benchmarks, not as a remedy for unstable continuation.
``benchmarks/gw_bse_matched_scaling.py --unchecked-ac`` exposes that choice
and ``--correct-pyscf-pade`` explicitly applies and labels the reference
PySCF 2.14 evaluator correction described above. It does not modify the
installed PySCF package. Correcting that evaluator alone does not remove
ill-conditioning of the fits.

Saving BSE results and restarting
--------------------------------

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
-----------------------------------

GW and BSE use the same public naming: ``save(path)`` exports results,
``checkpoint(directory)`` writes restart state, and the class method
``from_checkpoint(directory)`` creates a solver from it. The latter is a
constructor, rather than a mutation of an existing solver.

``gw.save('gw.npz')`` stores available QP energies, weights, residuals,
orbital indices, AC self-energy samples/frequencies, static corrections,
and continuation coefficients/nodes. Energies use Hartree and
``metadata_json`` records convergence, method and diagnostics. Partial or
empty results can be exported after failure; inspect the metadata.

For molecular factorized AC GW::

    gw.checkpoint('gw-checkpoint')
    gw = GW.from_checkpoint('gw-checkpoint', threads=8)
    gw.refit(ac='pade')       # reuse saved self-energy; no integration
    # gw.refit(ac='twopole') # alternate continuation of the same samples
    bse = BSE(gw, threads=8)

The checkpoint includes MO RI factors, reference orbital data, available
static screening, sampled self-energy and fit/QP results. It reuses the
pickle-free directory format and memory-mapped arrays used by BSE. The
original SCF/molecule driver and active quadrature iteration are not restored;
``run()`` on a restored GW object raises with instructions to use ``refit()``
or construct a fresh GW driver. Refitting retains the existing AC method's
stability checks and approximations; it introduces no new approximation.

Default QP-gap BSE builds its own static screening from the restored QPs.
Explicit ``screening_energies='hf'`` can reuse the saved matching GW static
screening without rebuilding it. Contour-deformation, spectral GW and
contour-repaired AC checkpoints are currently unsupported; their available
QP results can still be exported with ``save()``. Checkpoint creation requires
saved AC samples and static corrections, and does not claim to resume an
interrupted integration.
