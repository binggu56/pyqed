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
