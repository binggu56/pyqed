Geometric Quantum Dynamics
==========================

In PyQED, geometric quantum dynamics currently refers to the LDR/LDRFG solver
track: nuclear wavepacket propagation in locally diabatic electronic frames
where phases, overlaps, Berry/derivative-coupling information, and coordinate
metrics affect the dynamics.  The liquid-phase workflow described below is the
current end-to-end driver for this solver path.

Core Idea
---------

The Born-Oppenheimer electronic problem defines adiabatic states
``Phi_I(R)`` and energies ``E_I(R)`` at each nuclear geometry ``R``:

.. math::

   H_e(R) |\Phi_I(R)\rangle = E_I(R)|\Phi_I(R)\rangle.

Because the electronic basis depends on ``R``, nuclear motion sees geometric
terms. The derivative coupling vector is

.. math::

   d_{IJ}(R) =
   \langle \Phi_I(R) | \nabla_R \Phi_J(R) \rangle.

Near conical intersections or avoided crossings, these couplings can dominate
the dynamics and make a purely adiabatic representation inconvenient.

Overlap-Based Local Diabatization
---------------------------------

Instead of differentiating electronic states directly, PyQED often uses
overlaps between neighboring geometries:

.. math::

   A_{IJ}(R,R') =
   \langle \Phi_I(R) | \Phi_J(R') \rangle.

For small geometry steps, this overlap contains the same local geometric
information as derivative couplings:

.. math::

   A_{IJ}(R,R+\Delta R)
   =
   \delta_{IJ}
   + \Delta R \cdot d_{IJ}(R)
   + O(\Delta R^2).

The overlap matrix can be unitarized, usually by a polar/SVD factor, to define
a locally diabatic frame. In that frame, the electronic basis is transported
smoothly along the nuclear path and short-time propagation can avoid explicit
NAC vectors.

Locally Diabatic Representation
-------------------------------

Given a local unitary transformation ``U(R)``, the adiabatic potential matrix
is transformed into a diabatic-like matrix:

.. math::

   V^\mathrm{LD}(R)
   =
   U^\dagger(R) E^\mathrm{ad}(R) U(R),

where ``E_ad`` is diagonal in the adiabatic representation. Off-diagonal
elements in ``V_LD`` encode electronic coupling in the transported local basis.

The core LDR solver pieces are split across ``pyqed.ldr`` and
``pyqed.namd``:

* ``pyqed.ldr.ldr``: locally diabatic representation wavepacket dynamics.
* ``pyqed.ldr.qd``: quasi-diabatization from overlap matrices.
* ``pyqed.ldr.curvilinear`` and ``pyqed.ldr.curvilinear_2d``: curvilinear DVR
  propagation and overlap-modified kinetic propagation.
* ``pyqed.ldr.gwp``: Gaussian wavepacket basis and overlap utilities.
* ``pyqed.ldr.coarse_grained``: coarse-grained overlap-based dynamics.
* ``pyqed.namd.ldrfg``: LDRFG propagation with derivative-coupling/Berry
  structure.
* ``pyqed.namd.liquid_ldr``: liquid-driven and solvent-embedded LDR workflows,
  diagnostics, convergence checks, and readiness gates.

Periodic SSH-Holstein Benchmark
--------------------------------

``pyqed.ldr.PeriodicSSHHolsteinGQD`` is the first periodic model benchmark for
the GQD solver.  It retains one Bloch sector of a two-sublattice chain and one
quantized optical coordinate ``Q``.  The electronic Hamiltonian is

.. math::

   h_k(Q) =
   \begin{pmatrix}
   \Delta + gQ & -\left[t_1(Q)+t_2(Q)e^{-ik}\right] \\
   -\left[t_1(Q)+t_2(Q)e^{ik}\right] & -\Delta-gQ
   \end{pmatrix},

with alternating SSH-modulated bonds

.. math::

   t_1(Q)=t+\delta+\alpha Q,
   \qquad
   t_2(Q)=t-\delta-\alpha Q.

The full local vibronic potential includes the harmonic phonon energy,

.. math::

   V_k(Q)=h_k(Q)+\frac{1}{2}\omega^2Q^2 I.

After diagonalizing ``V_k`` at every DVR point, neighboring electronic frames
define overlap links

.. math::

   A_{IJ}^{ab}
   =\langle\Phi_I(Q_a)|\Phi_J(Q_b)\rangle,
   \qquad
   U^{ab}=\operatorname{polar}\!\left(A^{ab}\right).

The GQD Hamiltonian is then

.. math::

   H^{\mathrm{GQD}}_{aI,bJ}
   =T_{ab}U^{ab}_{IJ}
   +\delta_{ab}\delta_{IJ}E_I(Q_a).

The benchmark independently constructs the exact diabatic DVR Hamiltonian

.. math::

   H^{\mathrm{dia}}
   =T_Q\otimes I
   +\sum_a |Q_a\rangle\langle Q_a|\otimes V_k(Q_a).

Because the complete two-state electronic space is retained, the two
Hamiltonians must satisfy

.. math::

   H^{\mathrm{GQD}}
   =\mathcal U^\dagger H^{\mathrm{dia}}\mathcal U,
   \qquad
   \mathcal U=\bigoplus_a \Phi(Q_a).

This makes the model a strict validation of overlap-dressed kinetic transport,
wavepacket propagation, and local electronic phase-gauge covariance.  Run the
calculation and generate its PDF/PNG diagnostics with

.. code-block:: bash

   PYTHONPATH=. python examples/ldr/periodic_ssh_holstein_gqd.py \
     --output /private/tmp/periodic_ssh_holstein_gqd.pdf

The script also writes an NPZ data bundle and a JSON validation summary.  This
benchmark is a fixed-``k`` periodic electron-phonon model; extending it to
finite phonon momentum, coupled Bloch sectors, and ab initio periodic
electronic scans is a subsequent step toward solid-state nonadiabatic
dynamics.

Finite-Momentum Extension
~~~~~~~~~~~~~~~~~~~~~~~~~

``pyqed.ldr.PeriodicSSHHolsteinMomentumGQD`` retains all two-sublattice
states on a commensurate ``N``-point Bloch mesh.  A real standing-wave phonon
has the cell profile

.. math::

   u_R(Q_q)=Q_q f_R,
   \qquad
   f_R=
   \frac{\cos(qR+\phi)}
   {\sqrt{N^{-1}\sum_{R'}\cos^2(qR'+\phi)}},
   \qquad
   q=\frac{2\pi m}{N}.

It modulates the local SSH bonds and Holstein energy according to

.. math::

   t_{1,R}=t+\delta+\alpha u_R,
   \qquad
   t_{2,R}=t-\delta-\alpha u_R,
   \qquad
   \epsilon_R=\Delta+g u_R.

The electronic Hamiltonian is first assembled in the periodic real-space
supercell,

.. math::

   H_e(Q_q)=
   -\sum_R\left[
   t_{1,R}a_R^\dagger b_R
   +t_{2,R}a_{R+1}^\dagger b_R
   +\mathrm{h.c.}\right]
   +\sum_R\epsilon_R
   \left(a_R^\dagger a_R-b_R^\dagger b_R\right),

and then transformed with

.. math::

   c_{R\sigma}
   =\frac{1}{\sqrt N}\sum_k e^{ikR}c_{k\sigma}.

This construction produces the finite-momentum selection rule directly:

.. math::

   \left\langle k'\left|
   \frac{\partial H_e}{\partial Q_q}
   \right|k\right\rangle\ne 0
   \quad\Longrightarrow\quad
   k'=k\pm q\pmod{2\pi}.

The complete ``2N``-state electronic space is diagonalized at every phonon
DVR point and propagated with the same overlap-dressed GQD Hamiltonian.  The
benchmark also constructs the independent diabatic supercell Hamiltonian and
compares wavefunctions and momentum populations throughout the propagation.
Run it with

.. code-block:: bash

   PYTHONPATH=. python \
     examples/ldr/periodic_ssh_holstein_finite_q_gqd.py \
     --ncells 4 --q-index 1 \
     --output /private/tmp/periodic_ssh_holstein_finite_q_gqd.pdf

The generated NPZ bundle contains ``P_k(t)``, band-resolved momentum
populations, the phonon density, coupling-block norms, and exact-reference
errors.  This remains a one-electron, one-standing-wave benchmark.  A
production solid-state method still needs occupied many-electron states,
multiple phonon coordinates, an ab initio periodic scan provider, and a
controlled electronic active-space truncation.

Continuum-Embedded Projector GQD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A discrete set of isolated Born-Oppenheimer surfaces is not generally a
controlled electronic representation for a solid.  In the thermodynamic
limit, band, electron-hole, and excitonic scattering states form continua.
``pyqed.ldr.FeshbachEmbedding`` provides the general projected-resolvent
engine.  Its ``from_ldr`` adapter partitions a complete overlap-dressed GQD
Hamiltonian with an active projector :math:`P` and its complement
:math:`Q=1-P`:

.. math::

   H =
   \begin{pmatrix}
   H_{PP} & H_{PQ} \\
   H_{QP} & H_{QQ}
   \end{pmatrix}.

Eliminating the continuum gives the retarded active-space Green function

.. math::

   G_P^R(E)
   =
   \left[
   E+i\eta-H_{PP}-\Sigma^R(E)
   \right]^{-1},

with the Feshbach self-energy

.. math::

   \Sigma^R(E)
   =
   H_{PQ}
   \left[E+i\eta-H_{QQ}\right]^{-1}
   H_{QP}.

The corresponding continuum hybridization is

.. math::

   \Gamma(E)
   =
   i\left[\Sigma^R(E)-\Sigma^R(E)^\dagger\right]
   =-2\operatorname{Im}\Sigma^R(E),

and the active projected spectrum is

.. math::

   A_P(E)
   =
   -\frac{1}{\pi}
   \operatorname{Im}\operatorname{Tr}G_P^R(E).

``FeshbachEmbedding.from_ldr`` performs this partition *after* constructing
the complete geometric kinetic operator.  Consequently, :math:`H_{PQ}` retains
the active-to-continuum coupling caused by electronic-frame rotation along
the nuclear coordinates.  If the active overlap block on a neighboring-grid
edge is

.. math::

   S_P^{ab}
   =
   P_a U^{ab}P_b,

its singular values are the cosines of the principal angles between the two
active subspaces.  Values below one measure projector leakage into the
eliminated sector.  Unitarizing :math:`S_P^{ab}` while discarding
:math:`H_{PQ}` would remove this physical coupling; the embedded construction
instead retains it through :math:`\Sigma^R(E)`.

Two continuum backends are provided.  ``MatrixElectronicContinuum`` evaluates
the resolvent of a finite dense or sparse :math:`H_{QQ}`.  The diagonal
quadrature backend ``DiagonalElectronicContinuum`` evaluates

.. math::

   \Sigma^R(E)
   =
   \sum_c
   \frac{w_c V_cV_c^\dagger}
        {E+i\eta-\epsilon_c},

where :math:`\epsilon_c`, :math:`w_c`, and :math:`V_c` may eventually be
supplied by Brillouin-zone integration, GW/BSE electron-hole states, or an
interpolated band continuum.  It also exposes the exact finite-bath memory
kernel

.. math::

   K(t)
   =
   \sum_c w_c V_cV_c^\dagger e^{-i\epsilon_ct}.

The implementation is an exact Feshbach reduction for the finite Hamiltonian
or quadrature supplied to it, following H. Feshbach, *Annals of Physics* 5,
357-390 (1958), DOI `10.1016/0003-4916(58)90007-1
<https://doi.org/10.1016/0003-4916(58)90007-1>`_.  Its use with overlap-link
GQD is an adaptation.  It does not yet construct a thermodynamic-limit
continuum, dynamical electronic screening, multiphonon bath, or scalable
time-domain tensor-network propagation.  Exact factorization has separately
established a first-principles nonadiabatic and geometric framework for
periodic solids; see *Nonadiabaticity from First Principles:
Exact-Factorization Approach for Solids*, *Physical Review B* 112, 075102
(2025), `APS article
<https://journals.aps.org/prb/abstract/10.1103/dmpv-zqdh>`_.

Run the finite-continuum validation with

.. code-block:: bash

   PYTHONPATH=. python \
     examples/ldr/periodic_ssh_holstein_continuum_gqd.py \
     --output /private/tmp/periodic_ssh_holstein_continuum_gqd.pdf

The embedded projected spectrum is compared against the active block of the
complete two-surface GQD resolvent.  The comparison is an exact finite-model
identity and tests the embedding implementation; it is not evidence that the
SSH-Holstein model reproduces a material electronic continuum.

Half-Filled Independent Mode Scans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pyqed.ldr.PeriodicSSHHolsteinHalfFilledScan`` constructs the complete real
normal-mode basis of the supercell.  For four cells, the profiles are

.. math::

   Q_0,
   \qquad
   Q_{\pi/2}^{\cos},
   \qquad
   Q_{\pi/2}^{\sin},
   \qquad
   Q_\pi,

and satisfy

.. math::

   \frac{1}{N}\sum_R
   f_{\lambda R}f_{\mu R}
   =\delta_{\lambda\mu}.

Each scan varies one coordinate while holding the other three at zero.  This
requires only four one-dimensional electronic scans; it does not construct an
``npts**4`` direct-product nuclear grid.

The current benchmark is spinless.  Half filling of the four-cell,
two-sublattice supercell therefore means

.. math::

   N_{\mathrm{orb}}=2N=8,
   \qquad
   N_e=N=4,
   \qquad
   N_{\mathrm{det}}=\binom{8}{4}=70.

At each displacement, the scanner diagonalizes the one-particle Hamiltonian,

.. math::

   h(Q_\lambda)C_n(Q_\lambda)
   =\epsilon_n(Q_\lambda)C_n(Q_\lambda),

and generates every noninteracting determinant energy from its occupied
orbital set,

.. math::

   E_{\mathcal I}(Q_\lambda)
   =\sum_{n\in\mathcal I}\epsilon_n(Q_\lambda).

The instantaneous many-body excitations shown in the scan figure are

.. math::

   \Omega_I(Q_\lambda)
   =E_I(Q_\lambda)-E_0(Q_\lambda).

For this independent-electron Hamiltonian, the first excitation equals the
fundamental particle-hole gap,

.. math::

   \Omega_1
   =\epsilon_{N_e+1}-\epsilon_{N_e}.

Run all mode scans and write the PDF/PNG figure, complete NPZ arrays, and JSON
summary with

.. code-block:: bash

   PYTHONPATH=. python \
     examples/ldr/periodic_ssh_holstein_half_filled_scan.py \
     --output /private/tmp/periodic_ssh_holstein_half_filled_scan.pdf

The NPZ bundle retains all 70 determinant energies, all 16 single
particle-hole energies, orbital momentum weights, harmonic vibronic surfaces,
and mode profiles.  These are independent-particle excitations, not excitons;
electron-electron interactions require a correlated electronic solver or a
fermionic MPS representation.

Liquid-Phase LDR Workflow
-------------------------

PyQED also includes a liquid-phase LDR workflow in ``pyqed.namd.liquid_ldr``.
It separates the solvent trajectory from the quantum propagator: the liquid
configuration is converted into a solvent collective coordinate, and that
coordinate drives an LDR/LDRFG Hamiltonian through energy shifts and geometric
transport.  This is useful for testing how solvent fluctuations modulate
nonadiabatic populations before committing to a more expensive ab initio
embedded calculation.

The smoke workflow is ``examples/namd/liquid_phase_ldr.py``.  A compact
analytic run that generates a methanol-water trajectory, compares liquid and
static references, and audits Berry/no-Berry geometric effects is:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr.py \
     --md-steps 4 \
     --frames 4 \
     --waters 4 \
     --x-points 5 \
     --ldr-substeps auto \
     --ldr-substep-convergence 1,2,4 \
     --geometric-stride-convergence 1,2 \
     --geometric-gauge-check \
     --geometric-gauge-substeps auto \
     --geometric-gauge-substep-convergence 1,2,4 \
     --require-liquid-ldr-readiness \
     --output-dir /private/tmp/pyqed_liquid_phase_ldr

When ``--ldr-substeps auto`` or ``--geometric-gauge-substeps auto`` is used,
the workflow starts from the requested convergence candidates and can refine by
doubling the largest substep count until the relevant readiness check passes or
the corresponding ``--ldr-substep-auto-max`` /
``--geometric-gauge-substep-auto-max`` limit is reached.  This keeps strict
gauge/readiness runs from failing just because the initial ``1,2,4`` candidate
list was too short.

Important analytic artifacts are:

* ``liquid_phase_ldr_result.npz``: populations, solvent coordinate,
  Berry/no-Berry populations, step scores, and diagnostics.
* ``liquid_ldr_frame_diagnostics.csv``: one row per sampled liquid frame with
  solvent coordinate, q-dot, gap, Berry norm, geometric speed, energies, norms,
  and state populations.
* ``liquid_ldr_geometric_steps.csv``: per-interval geometric population
  diagnostics and solvent-driver correlations.
* ``liquid_ldr_geometric_hotspots.json`` and
  ``liquid_ldr_geometric_hotspot.xyz``: ranked liquid intervals where the
  geometric contribution is largest, including normalized driver attribution
  for solvent-coordinate jumps, geometric speed, and inverse energy gap.  The
  summary/report also aggregate these hot spots to identify the dominant liquid
  driver across the ranked geometric events.
* ``liquid_ldr_hotspot_driver_summary.json``: standalone aggregate of the
  dominant liquid drivers across ranked Berry hot spots, suitable for
  downstream scans without loading the full ``summary.json``.
* ``liquid_ldr_geometric_readiness.json``: machine-readable readiness verdict
  combining quality, substep convergence, gauge checks, and stride checks.
* ``readiness_summary.json``: compact liquid and embedded readiness status for
  downstream gating without reading the full ``summary.json``.
* ``liquid_ldr_geometric_population.json``: compact Berry/no-Berry population
  evidence bundle for downstream analysis.
* ``liquid_ldr_run_summary.csv``: one-row scalar summary for batch scans,
  including readiness, geometric population change, hot-spot driver, and key
  artifact paths.
* ``run_metadata.json``: exact CLI invocation, normalized options, readiness
  verdicts, and declared artifact paths for reproducibility.
* ``liquid_ldr_geometric_report.md`` and ``artifact_manifest.json``: human
  report and complete artifact inventory with file sizes, SHA-256 hashes, and
  absolute paths for cwd-independent verification.  The manifest also records
  a schema identifier and hash algorithm so downstream checks can reject
  incompatible bundles.

The manifest can be checked later without rerunning dynamics:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr.py \
     --verify-artifact-manifest /private/tmp/pyqed_liquid_phase_ldr/artifact_manifest.json \
     --verification-report /private/tmp/pyqed_liquid_phase_ldr/verification_report.json

To inspect an existing bundle and summarize readiness, compact limited-status
reasons, manifest integrity, diagnostic artifact paths, and the dominant
hot-spot driver without rerunning dynamics:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr.py \
     --inspect-bundle /private/tmp/pyqed_liquid_phase_ldr \
     --inspection-report /private/tmp/pyqed_liquid_phase_ldr/bundle_inspection.json

For repeated solvent realizations, ``examples/namd/liquid_phase_ldr_scan.py``
can run several seeded analytic liquid LDR jobs and aggregate their
``liquid_ldr_run_summary.csv`` rows into scan-level CSV/JSON artifacts:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr_scan.py \
     --seeds 31,32,33 \
     --md-steps 4 \
     --frames 4 \
     --waters 4 \
     --x-points 5 \
     --geometric-gauge-check \
     --geometric-gauge-tolerance 1e-3 \
     --geometric-gauge-substeps auto \
     --geometric-gauge-substep-convergence 1,2,4 \
     --require-liquid-ldr-readiness \
     --scan-plot \
     --output-dir /private/tmp/pyqed_liquid_phase_ldr_scan

The scan driver writes ``liquid_ldr_scan_summary.csv``,
``liquid_ldr_scan_summary.json``, ``liquid_ldr_scan_report.md``,
``liquid_ldr_scan_evidence.json``, ``liquid_ldr_scan_metadata.json``,
optional ``liquid_ldr_scan_summary.png``, and
``liquid_ldr_scan_artifact_manifest.json``.  These aggregate readiness counts,
quality counts, dominant hot-spot-driver counts, per-run artifact-manifest
verification status, an ensemble ``scan_readiness`` verdict, and
population/hot-spot score ranges, including median, standard deviation, and
relative standard deviation.  They also report how many runs exceed
``--min-geometric-signal`` and whether the same dominant hot-spot driver
recurs across non-missing runs.  If ``--max-signal-relative-stdev`` is set,
the scan readiness gate can also reject ensembles whose geometric signal
magnitude varies too much across solvent seeds.  The run summaries also retain
the final Berry/no-Berry population-delta direction, allowing scans to report
state/sign consensus and, with ``--min-final-direction-consensus-fraction``,
reject ensembles whose final geometric population change points in inconsistent
state directions.  Driver and final-direction consensus records explicitly
report ties so a deterministic representative value is not mistaken for a
unique mechanism; the scan inspection command prints the same tie flags and
tied values for quick terminal audits.  The JSON/report and optional PNG rank
or visualize the strongest child runs by geometric population signal so the
dominant solvent realization can be inspected directly.  The compact
``liquid_ldr_scan_evidence.json`` artifact keeps the scan verdict, thresholds,
signal statistics, driver consensus, top runs, and artifact pointers together
for downstream gating or citation without carrying child process logs.  The
``liquid_ldr_scan_metadata.json`` artifact records the top-level scan command,
normalized options and thresholds, child Python executable, child run commands,
stdout/stderr paths, and aggregate artifact paths for reproducibility.  Add
``--require-scan-readiness`` when the aggregate should fail unless the
requested ``--min-scan-count``, ``--min-ready-fraction``,
``--min-manifest-ok-fraction``, ``--min-signal-fraction``, optional
``--max-signal-relative-stdev``, ``--min-driver-consensus-fraction``, and
``--min-final-direction-consensus-fraction`` thresholds pass.  Add
``--require-verified-manifests`` when every child run's manifest must verify
regardless of the scan-readiness thresholds.  The driver can also aggregate
existing runs without launching dynamics:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr_scan.py \
     --input-dir /private/tmp/pyqed_liquid_phase_ldr_seed31 \
     --input-dir /private/tmp/pyqed_liquid_phase_ldr_seed32 \
     --output-dir /private/tmp/pyqed_liquid_phase_ldr_scan

The scan artifact manifest can be checked later without rerunning any child
dynamics:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr_scan.py \
     --verify-scan-artifact-manifest /private/tmp/pyqed_liquid_phase_ldr_scan/liquid_ldr_scan_artifact_manifest.json \
     --scan-verification-report /private/tmp/pyqed_liquid_phase_ldr_scan/scan_verification_report.json

To inspect a completed scan bundle, verify its aggregate artifacts, and report
why the ensemble is ready or limited:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr_scan.py \
     --inspect-scan-bundle /private/tmp/pyqed_liquid_phase_ldr_scan \
     --scan-inspection-report /private/tmp/pyqed_liquid_phase_ldr_scan/scan_inspection_report.json

For a cheap embedded end-to-end check, the same script can build a solvent
embedded H2 CASCI LDR trajectory, compute frame-overlap diagnostics, run
transported propagation, and require embedded readiness:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr.py \
     --md-steps 4 \
     --frames 4 \
     --waters 4 \
     --x-points 5 \
     --embedded-trajectory \
     --embedded-trajectory-frames 2 \
     --embedded-frame-overlaps \
     --embedded-transported-propagation \
     --embedded-ldr-substeps auto \
     --embedded-ldr-substep-convergence 1,2 \
     --embedded-hotspots-top-k 1 \
     --embedded-geometric-tolerance 1e-20 \
     --embedded-geometric-population-tolerance 1e-40 \
     --require-embedded-geometric-readiness \
     --output-dir /private/tmp/pyqed_embedded_liquid_ldr

The very small embedded smoke run uses permissive visibility thresholds because
the generated two-frame path can be geometrically quiet.  For production
analysis, use physically meaningful thresholds, more frames, and convergence
records for substeps, retained states, active space, and frame stride.

For the methanol reference workflow, ``--methanol-fg-audit-preset`` switches on
the embedded C-O CASCI LDR path, frame-overlap transport, transported
population geometry, embedded substep/stride checks, and embedded readiness.
This workflow keeps the methanol C-O stretch on the LDR/DVR grid while building
a single full-coordinate frozen-Gaussian path over the hydroxyl O-H stretch,
the C-O-H bend, and all solvent Cartesian coordinates in a methanol body frame.
It writes ``embedded_methanol_co_fg_path_diagnostics.json`` and stores FG
centers, momenta, masses, widths, labels, groups, source frames, Gaussian
overlap magnitudes, and width-scaled displacements in
``embedded_methanol_co_casci_ldr_trajectory.npz``.  The same preset also runs
an audit-level coupled LDRFG TDVP propagation of ``C,Q,P`` using a
path-linearized embedded electronic force and a trajectory-derived classical
force estimate from finite-difference FG momenta.  This produces
``embedded_methanol_co_ldrfg_tdvp_diagnostics.json`` plus ``tdvp_fg_*`` arrays
in the NPZ file.  The TDVP leg exercises the coupled equations, but it is not
yet a production arbitrary-displacement QM/MM gradient engine:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/liquid_phase_ldr.py \
     --methanol-fg-audit-preset \
     --md-steps 4 \
     --frames 4 \
     --waters 4 \
     --x-points 5 \
     --output-dir /private/tmp/pyqed_methanol_fg_liquid_ldr

Readiness Gates
---------------

The liquid workflow reports readiness instead of relying on a single green
plot.  The analytic gate checks whether:

* the Berry/no-Berry population signal is visible,
* norm drift is controlled,
* enough liquid intervals were sampled,
* requested substep convergence is stable,
* requested gauge invariance checks are stable,
* requested stride convergence preserves the geometric signal.

Embedded readiness additionally checks frame transport quality, transported
population geometry, and optional state, active-space, and frame-step
convergence.  A ``ready`` verdict means the diagnostics are internally
consistent for the sampled trajectory and thresholds.  It does not by itself
prove that the solvent trajectory is equilibrated or that the electronic
structure model is converged.

Curvilinear Coordinates
-----------------------

In internal or curvilinear coordinates ``q``, the nuclear kinetic energy uses a
coordinate metric:

.. math::

   T =
   \frac{1}{2}
   \sum_{\alpha\beta}
   p_\alpha G_{\alpha\beta}(q) p_\beta.

For fixed-angle triatomic coordinates, for example, the G-matrix contains
off-diagonal kinetic couplings between bond coordinates. PyQED's curvilinear
LDR implementation combines this kinetic operator with electronic-state
overlap matrices in split-operator propagation.

The coordinate chart and nuclear grid are separate:

.. code-block:: python

   from pyqed.ldr import Coord, LDR, keo

   coord = Coord(
       to_cartesian=geometry,
       bounds=((r1_min, r1_max), (r2_min, r2_max)),
   )

   ldr = LDR(
       mc,
       grid=grid,
       coord=coord,
       states=(1, 2),
       keo=keo.podolsky(),
   ).build()

Here ``mc.mol`` supplies the atomic masses. ``keo.podolsky()`` differentiates
``geometry(q)`` with JAX, samples the inverse vibrational metric and the
Podolsky pseudopotential, and builds active-axis MPO components without a
global nuclear matrix. Set ``pseudopotential=False`` or ``None`` to omit the
pseudopotential, or pass its sampled grid values explicitly. This is an
adaptation of Eq. (21) in E. Mátyus,
G. Czakó, and A. G. Császár, *J. Chem. Phys.* **130**, 134112 (2009),
`doi:10.1063/1.3076742 <https://doi.org/10.1063/1.3076742>`_. The automatic
path is limited to smooth, nonredundant, nonsingular molecular charts and
``J=0`` vibrational dynamics. It does not add constrained-coordinate,
linear-molecule, rotational, or Coriolis corrections, and the Cartesian map
must be JAX differentiable. Grid points must avoid coordinate singularities.

Tensor-Network Ab Initio LDR
----------------------------

The tensor-network driver separates the electronic sampling grid from the
nuclear dynamics DVR:

.. code-block:: python

   from pyqed.ldr import AbInitioFit, Coord, keo
   from pyqed.namd import TNLDR

   coord = Coord(
       to_cartesian=geometry,
       bounds=((r1_min, r1_max), (r2_min, r2_max)),
   )

   fit = AbInitioFit(
       mc,
       coord=coord,
       states=(1, 2),
   ).build()

   tnldr = TNLDR(
       fit,
       grid=dynamics_grid,
       coord=coord,
       keo=keo.podolsky(),
   ).build()

``coord`` supplies the shared generalized-coordinate convention, Cartesian
map, and fitting domain. ``dynamics_grid`` supplies only the nuclear basis and
KEO discretization. ``AbInitioFit`` always samples adaptively. It constructs a
Chebyshev candidate grid over the coordinate intervals, starts from
one-dimensional coordinate fibers through the anchor geometry, and adds
geometries selected by the current synchronized-feature defect and their
distance from previous samples. The candidate pool is bounded and the maximum
number of electronic-structure calls grows only linearly with the coordinate
dimension. The completed fit can be reused with multiple dynamics grids.

Links for the dynamics DVR are formed from continuous feature cores at
neighboring endpoints, so the electronic sampling nodes need not equal the
DVR nodes. Energy, feature, and KEO cores are combined directly into split MPO
components; no full electronic field, link grid, overlap fiber, or vibronic
Hamiltonian is evaluated on ``dynamics_grid``. Advanced studies that need a
custom sampling design can use ``AbInitioFit`` directly.

Every native electronic fit automatically creates a persistent
``ElectronicDatabase`` for the molecular identity. Each geometry is keyed by
its Cartesian coordinates and a protocol fingerprint containing the atomic
species, charge, spin, basis, electronic driver, mean-field method, active
space, roots, selected states, integral representation, and reference orbital
hashes. A repeated calculation therefore restores electronic frames and raw
overlap blocks before calling the quantum-chemistry scanner. The database path
and current-run statistics are available as ``tnldr.database_path`` and
``tnldr.database_info``. Set ``PYQED_ELECTRONIC_CACHE_DIR`` to place the
databases on project or cluster storage; otherwise PyQED uses the operating
system's persistent user-cache directory.

Born-Oppenheimer Hamiltonian Derivatives
----------------------------------------

``pyqed.qchem.geometric`` provides helper machinery for first- and second-order
Born-Oppenheimer Hamiltonian derivatives in an electronic-state basis. The main
entry point is ``bo_hamiltonian_derivatives``.

It builds Cartesian derivative tensors

.. math::

   F^a_{IJ}
   =
   \left\langle \Psi_I \left|
   \frac{\partial H}{\partial R_a}
   \right| \Psi_J \right\rangle,

.. math::

   G^{ab}_{IJ}
   =
   \left\langle \Psi_I \left|
   \frac{\partial^2 H}{\partial R_a \partial R_b}
   \right| \Psi_J \right\rangle,

and can project them onto normal-mode or coarse-grained coordinates. These
terms are useful for constructing vibronic model Hamiltonians and local
quadratic expansions around a reference geometry.

Example structure:

.. code-block:: python

   from pyqed.qchem.geometric import bo_hamiltonian_derivatives

   # state_model is a CASCI/CASSCF-like object with state RDM/TDM access.
   derivs = bo_hamiltonian_derivatives(
       state_model,
       state_ids=[0, 1],
       mode_vectors=normal_modes,
   )

   F = derivs.F_projected
   G = derivs.G_projected

Connection to Ehrenfest Dynamics
--------------------------------

The overlap-based Ehrenfest path in :doc:`tddft_ehrenfest` is the classical
trajectory analogue of local-diabatic propagation. At each nuclear step, the
driver computes an electronic state overlap between ``R_t`` and
``R_{t+dt}``, unitarizes that overlap, and propagates electronic amplitudes in
the transported basis.

This is useful when:

* analytic NACs are unavailable,
* finite-difference NACs are noisy,
* state phases flip between adjacent geometries,
* states exchange character near crossings,
* a local diabatic picture is more stable than an adiabatic one.

Typical Workflow
----------------

An overlap-based geometric dynamics calculation usually follows this pattern:

1. Choose electronic states and a nuclear coordinate grid or trajectory.
2. Compute adiabatic energies on each geometry.
3. Compute electronic-state overlaps between neighboring geometries.
4. Unitarize overlaps to define local transport.
5. Build local diabatic potential matrices or overlap-modified propagators.
6. Propagate wavepackets or Ehrenfest trajectories.
7. Monitor norm, population transfer, and phase continuity.

Examples
--------

Relevant examples:

* ``examples/ldr/ldr.py``
* ``examples/ldr/abinitio.py``
* ``examples/ldr/abinitio_pyscf.py``
* ``examples/ldr/overlap_matrix_approximation_2D.py``
* ``examples/ldr/overlap_matrix_approximation_Ndimension.py``
* ``examples/ldr/h3/1scan_PES_H3+.py``
* ``examples/ldr/h3/2calculate_overlap_nearest_neighbor.py``
* ``examples/namd/abinitio_ehrenfest_pyscf.py``
* ``examples/namd/liquid_phase_ldr.py``
* ``examples/namd/lif_population_dynamics.py``
* ``examples/qchem/bo_hamiltonian_derivatives.py``
* ``examples/qchem/bo_hamiltonian_derivatives_normal_modes.py``

Related Pages
-------------

* :doc:`theory`
* :doc:`dvr`
* :doc:`pyqed.namd`
* :doc:`tddft_ehrenfest`
* :doc:`qchem`
