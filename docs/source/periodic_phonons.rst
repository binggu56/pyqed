Periodic Phonons
================

PyQED provides an experimental, dependency-free finite-displacement phonon
driver.  It accepts forces from a caller-supplied calculator and includes a
native all-electron or GTH Gamma-point KRHF force calculator.  Explicit GTH
data require no external electronic-structure package.

Native KRHF Forces
------------------

For a converged closed-shell KRHF reference at fixed lattice, the implemented
energy derivative is

.. math::

   \frac{dE}{dR_A}
   = \operatorname{Tr}\!\left[P h^{(A)}\right]
   + \frac{1}{2}\operatorname{Tr}\!\left[P G^{(A)}(P)\right]
   - \operatorname{Tr}\!\left[W S^{(A)}\right]
   + \frac{dE_{\mathrm{Ewald}}^{NN}}{dR_A},

where

.. math::

   W_{\mu\nu}
   = \sum_{i\in\mathrm{occ}}
     f_i\epsilon_i C_{\mu i}C_{\nu i}^{*}.

The first term contains analytic derivatives of the Bloch-summed kinetic and
Ewald electron--ion matrices.  For GTH cells it also differentiates the local
Gaussian correction and the separable nonlocal projectors,

.. math::

   V^{\mathrm{NL}}_{\mu\nu}
   = \sum_{I,lm,ij}
     B^{I,lmi*}_{\mu}\,h^I_{l,ij}\,B^{I,lmj}_{\nu}.

The second term contains analytic Gaussian-pair Fourier derivatives for
reciprocal J/K.  These derivatives use the same compiled, screened periodic
pair-FT plan as the SCF build and are contracted in G-vector blocks without an
AO four-index derivative tensor.  With ``jk_builder="ewald"`` the
all-electron route instead includes the short-range four-center derivative and
Ewald background term.  The third term is the Pulay contribution.  The
Madelung exchange correction is differentiated through both ``pair_cut``
overlap factors.  The ionic term differentiates the real- and reciprocal-space
Ewald sums directly.

Automatic Reciprocal Domain
---------------------------

Direct KRHF calculations use ``recip_cut="auto"`` by default.  The reciprocal
domain is selected from the actual compiled periodic AO-pair Fourier plan,
rather than from a fixed integer shell.  For reciprocal shell
:math:`\mathcal S_n`, the electronic bound is

.. math::

   B_n^{ee}
   = \max_{k,\mu\nu}
     \sum_{\mathbf G\in\mathcal S_n}
     \frac{4\pi}{\Omega G^2}
     d_{\mathbf G}
     \left|F_{\mu\nu}^{k}(\mathbf G)\right|^2,

where :math:`d_{\mathbf G}=1` for ``jk_builder="reciprocal"`` and
:math:`d_{\mathbf G}=\exp[-G^2/(4\eta^2)]` for the Ewald J/K builder.  The
electron--nuclear and ion--ion pieces are bounded by

.. math::

   B_n^{eN}
   = \max_{k,\mu\nu}
     \sum_{\mathbf G\in\mathcal S_n}
     \frac{4\pi}{\Omega G^2}
     e^{-G^2/(4\eta^2)}
     \left(\sum_A |Z_A|\right)
     \left|F_{\mu\nu}^{k}(\mathbf G)\right|,

.. math::

   B_n^{NN}
   = \sum_{\mathbf G\in\mathcal S_n}
     \frac{2\pi}{\Omega G^2}
     e^{-G^2/(4\eta^2)}
     \left(\sum_A |Z_A|\right)^2.

With :math:`B_n=B_n^{ee}+B_n^{eN}+B_n^{NN}`, the omitted tail is estimated
from the largest of the last three shell ratios,

.. math::

   r_n = \max_{j=n-2,n-1,n}\frac{B_j}{B_{j-1}},
   \qquad
   T_n = B_n\frac{r_n}{1-r_n}.

The first shell satisfying :math:`T_n\leq\texttt{recip_precision}` is selected.
For a k-point mesh, the final integer cutoff also includes the largest
reciprocal-lattice transfer needed by :math:`\mathbf k_j-\mathbf k_i`.
``recip_max_cut`` bounds the search and raises an error if the target is not
reached.  The resolved cutoff and tail estimate are available through
``mean_field.recip_auto_info`` and ``mean_field.integral_build_timings``.
This controls the direct KRHF Ewald/reciprocal domain; GDF retains its separate
``gdf_precision``, ``gdf_omega``, and ``gdf_mesh`` controls.

For full-Coulomb and range-separated GDF, the implementation differentiates
the raw three-center tensor :math:`B` and auxiliary metric :math:`M`.  In the
range-separated path these tensors have the compensated form

.. math::

   M = M^{\mathrm{LR}}_{\mathbf G}
       + M^{\mathrm{SR}}_{\mathbf R}
       - g_0\,\mathbf q_{\mathrm{aux}}\mathbf q_{\mathrm{aux}}^\dagger,

.. math::

   B = B^{\mathrm{LR}}_{\mathbf G}
       + B^{\mathrm{SR}}_{\mathbf R}
       - g_0\,\mathbf q_{\mathrm{aux}} S,
   \qquad
   g_0 = \frac{\pi}{\omega^2\Omega}.

The analytic response follows the same smooth/compact auxiliary-shell and
AO-pair partitions as the SCF builder.  It contracts the result through the
retained metric pseudoinverse :math:`R=M^+`, including

.. math::

   R^{(A)} = -R M^{(A)} R

for a full-rank metric, with the corresponding spectral pseudoinverse
derivative when linearly dependent auxiliary modes have been removed.  This
avoids a gauge-dependent derivative of the whitened cderi factors.

CPHF-Relaxed Gamma Hessian
--------------------------

For ``jk_builder="reciprocal"`` or ``"ewald"``, a converged Gamma-point
KRHF reference exposes ``mean_field.Hessian()``.  The driver forms the explicit
Fock perturbations

.. math::

   F^{[1]}_x = h^{[1]}_x + G^{[1]}_x[P^{(0)}],

solves all :math:`3N` moving-basis CPHF equations together, and assembles the
relaxed Hessian response

.. math::

   H^{\mathrm{resp}}_{xy}
   = \operatorname{Tr}\!\left[P^{(y)}h^{[1]}_x\right]
   + \frac{1}{2}\operatorname{Tr}\!\left[P^{(y)}G^{[1]}_x[P^{(0)}]\right]
   + \frac{1}{2}\operatorname{Tr}\!\left[P^{(0)}G^{[1]}_x[P^{(y)}]\right]
   - \operatorname{Tr}\!\left[W^{(y)}S^{[1]}_x\right].

The analytic ionic Ewald Hessian and explicit electronic second derivative are
added to this response.  For an all-electron reciprocal J/K reference the
explicit contribution is

.. math::

   H^{\mathrm{explicit}}_{xy}
   = \operatorname{Tr}\!\left[P^{(0)}h^{[2]}_{xy}\right]
   + \frac{1}{2}\operatorname{Tr}\!\left[
       P^{(0)}G^{[2]}_{xy}[P^{(0)}]\right]
   - \operatorname{Tr}\!\left[W^{(0)}S^{[2]}_{xy}\right]
   + H^{\mathrm{NN}}_{xy}.

The overlap, one-electron, reciprocal Coulomb/exchange, and ionic Ewald second
derivatives in this expression are evaluated by native analytic kernels.  The
calculation therefore requires one reference SCF and no coordinate step.  The
one-electron implementation evaluates rectangular derivative-basis blocks for
second-left/base, base/second-right, and first-left/first-right Gaussian pairs
with the compiled periodic kernel.  The reciprocal AO-pair derivatives use
the compiled periodic pair-FT plans, and the fixed-density J/K derivatives for
all CPHF response densities share one G-vector traversal.  Thus neither the
Gaussian integral work nor the geometry-dependent pair Fourier tensors are
rebuilt separately for each nuclear perturbation.  The Ewald J/K and
reciprocal GTH paths retain a validation/fallback backend that
central-differences analytic first derivatives without displaced SCF
calculations:

.. math::

   A^{[2]}_{xy}
   \simeq \frac{A^{[1]}_x(R_y+\delta)-A^{[1]}_x(R_y-\delta)}{2\delta}.

The mass-weighted Gamma dynamical matrix is

.. math::

   D_{A\alpha,B\beta}(\Gamma)
   = \frac{H_{A\alpha,B\beta}}{\sqrt{M_A M_B}}.

The API returns both Cartesian force constants and signed frequencies:

.. code-block:: python

   hessian = mean_field.Hessian()
   force_constants = hessian.kernel(second_derivative_backend="analytic")
   frequencies_cm1 = hessian.frequencies()

Both the analytic Gamma Hessian and ``FiniteDisplacementPhonon`` expose one
branch through ``mode(qpoint, branch)``.  The returned
``PeriodicPhononMode`` carries the signed frequency in atomic units and the
unit-norm eigenvector of the mass-weighted dynamical matrix,

.. math::

   \sum_{B\beta}D_{A\alpha,B\beta}(q)e_{B\beta}^{q\nu}
   =\omega_{q\nu}^{2}e_{A\alpha}^{q\nu},
   \qquad
   \sum_{A\alpha}|e_{A\alpha}^{q\nu}|^2=1.

Its ``cartesian_displacement`` property is

.. math::

   u_{A\alpha}^{q\nu}
   =\frac{e_{A\alpha}^{q\nu}}{\sqrt{M_A}}.

Fractional reciprocal coordinates are stored explicitly.  A deterministic
global phase is chosen for each eigenvector; only sums over a degenerate mode
subspace are gauge invariant when branches are exactly degenerate.

The default ``second_derivative_backend="auto"`` selects the analytic path
when available.  ``"finite_difference"`` keeps the derivative-difference
backend available for validation; only that backend uses ``step``.

Why a Supercell Is Needed
-------------------------

The primitive cell defines the atoms and therefore the :math:`3N` phonon
branches.  It does not by itself provide the real-space range of the force
constants.  Displacing an atom in a primitive cell with periodic boundary
conditions displaces every translated copy identically, so the perturbation
has phase

.. math::

   e^{i\mathbf q\cdot\mathbf R}=1,

and samples only :math:`\mathbf q=\Gamma`.  A supercell permits one translated
copy to be displaced independently.  The resulting forces resolve
:math:`\Phi_{i\alpha,j\beta}(\mathbf R)` over several lattice translations,
which can then be Fourier transformed to arbitrary commensurate
:math:`\mathbf q`.  Increasing the supercell converges the neglected
long-range tail of these force constants.  A primitive-cell DFPT solver could
avoid explicit supercells by solving the response separately at each
:math:`\mathbf q`.  PyQED now provides the :doc:`periodic_cphf` electronic
response for :math:`\mathbf k\to\mathbf k+\mathbf q`, including coupled
:math:`\pm\mathbf q` GDF response, and connects the Gamma response to the
analytic Hessian above.  Nonzero-q nuclear perturbation derivatives and
dynamical-matrix assembly are still required before this can replace finite
displacements for a complete phonon dispersion.

Force Constants
---------------

For a primitive-cell displacement :math:`u_{i\alpha}` and supercell force
:math:`F_{j\beta}(\mathbf R)`, central differences give

.. math::

   \Phi_{i\alpha,j\beta}(\mathbf R)
   = -\frac{\partial F_{j\beta}(\mathbf R)}{\partial u_{i\alpha}}
   \simeq -\frac{
       F_{j\beta}(\mathbf R;+\delta_{i\alpha})
       -F_{j\beta}(\mathbf R;-\delta_{i\alpha})
   }{2\delta}.

Only the atoms in one primitive reference cell are displaced.  Without
space-group reduction this requires :math:`6N_{\mathrm{atom}}` analytic-force
calculations.  Pair-interchange symmetry and the acoustic sum rule

.. math::

   \sum_{j\mathbf R}\Phi_{i\alpha,j\beta}(\mathbf R)=0

are imposed before reciprocal interpolation.

For fractional reciprocal coordinate :math:`\mathbf q`, the mass-weighted
dynamical matrix is

.. math::

   D_{i\alpha,j\beta}(\mathbf q)
   = \frac{1}{\sqrt{M_iM_j}}
     \sum_{\mathbf R}\Phi_{i\alpha,j\beta}(\mathbf R)
     e^{2\pi i\mathbf q\cdot\mathbf R}.

Its signed square-root eigenvalues are reported in atomic units or
:math:`\mathrm{cm}^{-1}`.  Negative frequencies denote unstable harmonic
modes.

Example
-------

The fast harmonic model exercises the complete supercell-to-spectrum path:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_phonon.py \
       --output /private/tmp/pbc_phonon_spectrum.pdf

The CPHF-relaxed Gamma-point KRHF Hessian and its diagnostic figure are
reproducible with

.. code-block:: console

   PYTHONPATH=. python examples/pbc_gamma_hessian.py \
       --output /private/tmp/pbc_gamma_hessian.pdf

The native phonon-to-exciton interface is exercised by

.. code-block:: console

   PYTHONPATH=. python examples/pbc_lih_exciton_phonon_convergence.py \
       --native-phonons --structure molecular --lattice-constant 7.0 \
       --meshes 2 --recip-cuts 2 --skip-validation

This periodic molecular-cell run selects a stable internal branch and is an
engine qualification; several low-frequency branches of the compact model
remain imaginary and are not a mechanically stable material reference.  The
rocksalt KRHF references tested at :math:`a=7.72\,a_0` have imaginary harmonic
modes and are rejected by the coupling driver.  A rocksalt material benchmark
therefore requires a mechanically stable electronic reference, expected to be
periodic DFT rather than the present KRHF force driver.  Converged material predictions
require supercell, displacement, force, k-mesh, q-mesh, orbital-basis, and
auxiliary-basis convergence.

PySCF Benchmark
---------------

PySCF does not currently expose an all-electron periodic Hessian, and its
all-electron periodic core-Hamiltonian gradient raises ``NotImplementedError``.
The cross-code benchmark therefore compares central second differences of
tightly converged total energies,

.. math::

   H_{x x}
   \simeq \frac{E(R_x+\delta)-2E(R_x)+E(R_x-\delta)}{\delta^2}.

Because this estimate has a leading :math:`O(\delta^2)` error, the benchmark
uses two steps and reports the Richardson zero-step estimate

.. math::

   H(0) \simeq \frac{r^2 H(\delta)-H(r\delta)}{r^2-1},
   \qquad r=2.

PySCF uses a converged FFTDF mesh, while PyQED uses the reciprocal J/K backend
with converged real-space AO-pair domains and a precision-selected reciprocal
domain.  Run the
benchmark and generate its JSON, PDF, and PNG outputs with

.. code-block:: console

   PYTHONPATH=. python examples/benchmark_pbc_gamma_hessian_pyscf.py \
       --output /private/tmp/pbc_gamma_hessian_pyscf_benchmark.pdf

For the periodic H2 STO-3G test at ``recip_precision=1e-8``, the direct PyQED
calculation selects reciprocal cut 10 automatically, with estimated tail
:math:`6.12\times10^{-9}`.  The analytic diagonal is

.. math::

   (H_{xx},H_{yy},H_{zz})_{mathrm{PyQED}}
   = (0.4539283128,-0.0256055402,-0.0256055402)\;E_h/a_0^2,

while the extrapolated PySCF FFTDF diagonal is

.. math::

   (H_{xx},H_{yy},H_{zz})_{mathrm{PySCF}}
   = (0.4539283278,-0.0256055520,-0.0256055527)\;E_h/a_0^2.

The maximum component difference is :math:`1.50\times10^{-8}\;E_h/a_0^2`,
and the total-energy difference is :math:`9.66\times10^{-10}\;E_h`.
On the benchmark machine the analytic Hessian took 23.28 seconds; timing is
machine dependent and the JSON output records the measured SCF and Hessian
times.

A native ab initio smoke calculation uses analytic KRHF forces throughout:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_phonon.py \
       --backend krhf --supercell-x 2 --recip-cut 2 \
       --output /private/tmp/pbc_h2_krhf_phonon.pdf

The same supercell workflow can use explicit GTH-Pade hydrogen data without
loading an external pseudopotential library:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_phonon.py \
       --backend krhf --gth-pade --supercell-x 2 --recip-cut 2 \
       --output /private/tmp/pbc_h2_gth_phonon.pdf

The programmatic API accepts any force calculator that returns forces in
Hartree/Bohr for ``(symbols, positions, lattice)``:

.. code-block:: python

   from pyqed.pbc import FiniteDisplacementPhonon

   phonon = FiniteDisplacementPhonon(
       cell,
       force_calculator,
       supercell=(3, 3, 3),
       displacement=0.01,
   ).run()
   bands = phonon.band_structure(
       [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]],
       labels=("Gamma", "X", "M"),
   )

Metals
------

Metallic calculations should use a force calculator with a converged
electronic temperature, k mesh, supercell, and displacement.  The static force
constants represent the adiabatic electronic response.  Modes near Kohn
anomalies or the zone-center collisionless limit may require an explicit
frequency-dependent electron-phonon self-energy beyond this solver.

Limitations
-----------

* Only three-dimensional cells are supported.
* The current implementation displaces every primitive atom and Cartesian
  direction; space-group displacement reduction remains future work.
* Native analytic KRHF forces currently support closed-shell Gamma-point
  cells.  All-electron calculations can use ``jk_builder="reciprocal"`` or
  ``"ewald"``.  GDF calculations support full-Coulomb and range-separated
  response, including compensated smooth/compact shell partitions.  GTH
  calculations can use ``"reciprocal"`` or ``"gdf"``.  Non-Gamma k-point
  forces and lattice stress are not implemented yet.
* The CPHF-relaxed Hessian is fully analytic for closed-shell, all-electron,
  Gamma-point reciprocal J/K references.  Ewald J/K and reciprocal GTH
  references use a central difference of analytic first derivatives.  GDF and
  nonzero-q Hessians are not connected yet.
* Long-range nonanalytic LO--TO corrections and electron-phonon linewidths are
  not included.
* Atomic positions must be relaxed and force constants must be converged with
  supercell size, k mesh, smearing, basis, pseudopotential, and displacement.
