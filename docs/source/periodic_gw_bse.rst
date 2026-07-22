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
       real_cut=0,
       pair_cut=0,
       recip_cut=1,
       jk_builder="ewald",
   ).run()

   gw = KGW(mf, eta=1e-3).g0w0(direct_scale=1.0)
   tda = KTDA(gw).run(q_index=0, direct_scale=1.0, nroots=2)
   bse = KBSE(gw).run(q_index=0, direct_scale=1.0, nroots=2)

   print(gw.e_qp)  # shape: (nkpts, nband)
   print(tda.e)
   print(bse.e)

The runnable repository example is:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_h2_gw_bse.py

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

The H2 benchmark in ``examples/pbc_h2_pyscf_gw_benchmark.py`` compares the
current native PyQED path against PySCF PBC KGW:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_h2_pyscf_gw_benchmark.py \
       --finite-size-correction

By default this is a diagnostic comparison rather than a strict
apples-to-apples benchmark.  PyQED's fast multi-k GW path uses the native
``reciprocal_ewald_lr`` transition factors, while PySCF KGW uses Gaussian
density fitting for the full Coulomb interaction.  The benchmark therefore
also records ``coulomb_metric_diagnostics`` in the JSON output.  Large
``relative_norm_delta`` values there indicate that the GW discrepancy is
already present in the bare transition Coulomb metric, before the RPA/GW
self-energy is built.

For the closest PySCF comparison, use the optional PySCF density-fitting
backend:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_h2_pyscf_gw_benchmark.py \
       --coulomb-component pyscf_gdf --finite-size-correction

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

The reciprocal mesh is estimated with a finite range-separation seed, avoiding
the prohibitively large full-Coulomb mesh that would otherwise be selected for
tight core Gaussians.  The same precision derives the short-range image box,
an integral-screening tolerance, and a relative auxiliary-metric threshold.
The latter defaults to ``sqrt(gdf_precision)`` so quadrature-noise modes are
not amplified by the inverse metric; set ``mf.gdf_metric_relative_tol``
explicitly when performing a rank-convergence study.

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
values.  Short-range image-pair tensors are streamed directly into all
requested Bloch blocks with bounded in-flight worker tasks and are never
stored in the image-component cache.  The final q-resolved AO/MO factors are
still retained, so their system-size-dependent storage is separate from this
workspace limit.  An explicit ``gdf_short_range_cut`` overrides the automatic
image box.  Setting a nonzero manual ``gdf_short_range_screen_tol`` remains a
diagnostic opt-in and also requires
``gdf_allow_heuristic_short_range_screening=True``.

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

``examples/pbc_si_bse_haydock_convergence.py`` provides a silicon k-mesh
driver and records operator-build time, recursion time, storage, roots, and
spectra.  Its default Gamma-centered meshes contain :math:`\Gamma` at every
mesh size.  Mixing odd Gamma-containing and even shifted Monkhorst--Pack grids
can otherwise produce a large parity oscillation in the apparent optical edge.

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
matrix-free path currently implements optical :math:`q=0` TDA.  Ordinary
:math:`q=0` BSE also does not include the phonon-assisted indirect absorption
edge of silicon.

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

The runnable LiH example writes CSV/NPZ data and PNG/PDF figures:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_lih_gw_pes.py --mesh 2 --backend gdf

PySCF 2.12 exposes periodic ``KRGWAC`` quasiparticle energies and the
imaginary-axis correlation self-energy, but not a real-axis PES driver.  The
independent benchmark therefore compares :math:`\Sigma^c(i\omega)`,
linearized quasiparticle poles, and spectra reconstructed with the same
two-pole analytic-continuation model:

.. code-block:: console

   PYTHONPATH=. python \
       examples/pbc_lih_gw_pyscf_spectral_benchmark.py --mesh 2 --nw 100

The benchmark reports selected-band errors separately, so bands omitted from
a PyQED target calculation are not accidentally compared to corrected PySCF
virtual energies.

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
* dense transition-space full BSE and finite-q TDA matrices, suitable for
  small cells; optical :math:`q=0` TDA also has a matrix-free Haydock path;
* no spin-orbit, unrestricted, finite-temperature, analytic-continuation, or
  force support;
* the default factorized Coulomb kernels use the reciprocal Ewald long-range
  component, not the full short-range plus reciprocal dense Ewald ERI;
* ``coulomb_component="full_ewald"`` is currently a dense small-cell native
  Ewald diagnostic for response kernels, diagonal GW self-energy, and BSE pair
  couplings, not a production large-cell algorithm;
* ``coulomb_component="gdf"`` is currently a native auxiliary-basis GDF backend
  with explicit finite image sums for the three-center tensors.  It removes the
  PySCF runtime dependency from the vector backend but still needs performance
  optimization and broader convergence validation for larger cells;
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
two-k H2 KRHF example path.  ``examples/pbc_h2_pyscf_gw_benchmark.py`` runs a
matching small-cell PySCF PBC KGW comparison and writes CSV/JSON diagnostics.
