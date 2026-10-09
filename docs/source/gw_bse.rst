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

BSE requires finite QP energies for all orbitals and rejects a partial
``orbs`` calculation. A matching contour reference supplies static screening
without populating ``bse.e_rpa`` or ``bse._M``; these remain None, rather than
holding artificial screening poles. Changed reference orbitals or screening
energies invalidate reuse. Spectral screening remains available for other
references and screening methods.

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

``BSE(gw)`` reuses the spatial screening poles and couplings when the
screening is TDH and its mean-field energies and MO coefficients match the
cached GW reference exactly. Changed-energy evGW screening is not reused
for mean-field BSE screening. Cached arrays are copied into the BSE object.
This avoids a second spatial RPA solve without changing the static kernel.

A separate numerical issue was exposed by the changed floating-point
summation order: the old unconstrained secant solve could select a
negative-weight quasiparticle crossing. Molecular diagonal Dyson roots
now use analytic self-energy derivatives and coupling continuation from
zero to full self-energy, checking positive residue and an absolute
1e-9 Hartree equation residual. Failed Newton steps trigger an outward
bracket search from the previous root, accepting the nearest found upward
crossing; the coupling step is reduced if no acceptable bracket is found.
At a branch fold this can select a neighboring positive-weight branch;
it is not a guarantee of uninterrupted analytic continuation. Failure raises
instead of silently replacing the result with a mean-field energy. ``qp_weights`` and ``qp_residuals``
record these checks for the last Dyson solve (before any evGW damping).
This is a numerical branch-selection adaptation of Hedin's Dyson equation, *Phys. Rev.* **139**, A796 (1965),
https://doi.org/10.1103/PhysRev.139.A796; it is not a complete satellite
solver and does not guarantee the root with largest spectral weight.

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

Nonsymmetric Davidson for molecular full BSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The matrix-free molecular full-BSE path defaults to compiled nonsymmetric
Davidson. SciPy Arnoldi remains available with ``eigensolver="arpack"``:

.. code-block:: python

   bse = BSE(gw).run(
       nroots=5, low_rank=True, eigensolver="davidson",
       tol=1e-9, max_cycle=200, max_space=40, batch_columns=8,
   )
   print(bse.e, bse.info["residual_norms"], bse.info["metric_error"])

The same controls are accepted by ``BSE.solve_bse`` and the module-level
``solve_bse``. ``max_space`` is Davidson's subspace cap or ARPACK's ``ncv``.
``batch_columns`` optionally applies the full A/B kernel to bounded blocks;
it requires Davidson. It is unset by default. Iterative controls require the
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

Together with the six cc-pVDZ molecule checks above, this supports making
Davidson the default for ``solve_bse`` and ``BSE.solve_bse``, and for the
matrix-free branch of ``BSE.run``. ``BSE.run(eigensolver=None)`` selects
the path's default; explicit iterative solver controls require
``low_rank=True`` (or automatic selection through pair factors). Dense
Casida and TDA dispatch are unchanged. Batching remains opt-in. No fallback
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
