CASCI and CASSCF
================

PyQED provides native active-space methods for multireference quantum
chemistry. CASCI optimizes CI coefficients in a fixed active orbital space.
CASSCF additionally optimizes the orbitals.

Direct CASCI Diagnostics
------------------------

The restricted real direct-CASCI solver uses a native single- or multiroot
Davidson implementation when the compiled backend and CBLAS are available.
Linux builds use the system BLAS and macOS builds use Accelerate. Other
Hamiltonian types use the tested Python Davidson fallback.

Every completed solve exposes structured status instead of silently changing
backends:

.. code-block:: python

   from pyqed.qchem import CASCI
   from pyqed.qchem.mcscf.direct_ci import direct_ci_capabilities

   print(direct_ci_capabilities())

   mc = CASCI(mf, ncas=6, nelecas=6).run(nstates=3, method="direct_ci")
   print(mc.converged)
   print(mc.direct_ci_diagnostics)
   print(mc.direct_ci_fallback_reason)

Native diagnostics include the iteration count, final residual norms, energy
changes, subspace dimension, root count, and whether a restart guess was used.
If native execution is unavailable or fails to converge,
``direct_ci_fallback_reason`` records the exact reason before the Python solver
is used.

For geometry scans, request coefficient reuse and overlap-based root homing:

.. code-block:: python

   scanner = mc.as_scanner(
       nstates=3,
       method="direct_ci",
       reuse_ci=True,
       root_homing=True,
       root_homing_cushion=2,
   )
   result = scanner(new_geometry)
   print(result.root_tracking_overlaps)

Root homing evaluates the biorthogonal electronic overlap between adjacent
geometries, performs a one-to-one maximum-overlap assignment, and phase-aligns
the selected CI vectors. Keep the default two-state Davidson root cushion near
dense manifolds and avoided crossings.

Basic CASSCF
------------

.. code-block:: python

   from pyqed.qchem import CASSCF, Molecule

   mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
   mol.build(eri="factors")

   mf = mol.RHF().run()
   mc = CASSCF(mf, ncas=2, nelecas=2).run()

   print(mc.e_tot)

``ncas`` is the number of active spatial orbitals. ``nelecas`` is the number
of active electrons, either as an integer for closed-shell active spaces or as
``(nalpha, nbeta)`` for explicit spin sectors.

Constrained Orbital Optimization (COCAS)
---------------------------------------

``COCAS`` alternates CASCI with orthonormality-constrained orbital
minimization at fixed RDMs. The default ``diis_residual="step"`` pairs each
orbital-map output with its displacement from the accepted base, using the
original absolute regularization. Candidates are aligned in redundant
orbital gauges before entering history. Exact CASCI discards active-active
gauge rotations; finite-bond DMRG retains them as physical variables.

.. code-block:: python

   from pyqed.qchem import COCAS

   mc = COCAS(mf, ncas=8, nelecas=8, optimizer="LBFGS",
              optimizer_max_steps=20, diis=True, diis_space=6,
              diis_start=2, orb_grad_tol=1e-4).run()

Two opt-in residual variants are available:

* ``diis_residual="transported_step"`` aligns all historical orbital-map
  displacements in a common redundant gauge, then projects them into the
  current physical tangent space.
* ``diis_residual="gradient"`` uses the physical gradient evaluated with
  each accepted post-CI state's RDMs, with the same gauge alignment and
  tangent transport. State averaging uses weighted RDMs.

The optional variants normalize the Gram matrix before regularization and
retry excessive weights with a shorter history. They are experiments, not
general convergence improvements: in the Fe(CO)5 def2-SVP CAS(8,8) CD test,
transported displacement did not reduce the six-step restart convergence,
while the raw gradient variant needed twelve steps. Both were worse than
the original displacement DIIS after a bounded 20-step RHF-start comparison,
so the original mode remains the default.

The inner L-BFGS optimizer transports every retained secant pair to the
current tangent space and discards pairs with nonpositive projected
curvature. ``physical_inner=True`` optionally restricts inner SD/RCG/L-BFGS
gradients, directions and transported history to physical orbital blocks.
It discards core-core rotations, and discards active-active rotations only
for exact CASCI; finite-bond DMRG retains active-active rotations. This is an
experimental restriction: active rotations are redundant for CI-relaxed
CAS energies, while frozen active RDMs generally change their energy under
such rotations. The restricted inner subproblem therefore differs from
unrestricted fixed-RDM minimization. It is not supported with AH or NEWTON.

The limited-memory recursion adapts J. Nocedal, *Updating quasi-Newton
matrices with limited storage*, Math. Comp. **35**, 773–782 (1980),
https://doi.org/10.1090/S0025-5718-1980-0572855-7. Polar retraction and
projected vector transport follow P.-A. Absil, R. Mahony and R. Sepulchre,
*Optimization Algorithms on Matrix Manifolds*, Princeton University Press
(2008), https://sites.uclouvain.be/absil/amsbook/. The manifold adaptation
does not inherit the Euclidean method's superlinear convergence guarantee.

``orbital_update="relaxed_lbfgs"`` selects one physical orbital step per CI
solve, using L-BFGS secants between accepted post-CI states. This uses the
observed change in the CI-relaxed gradient without solving explicit CI
response equations. Choose ``diis=False`` with this update. For example:

.. code-block:: python

   mc = COCAS(mf, ncas=8, nelecas=8, orbital_update="relaxed_lbfgs",
              diis=False, max_cycles=100, orb_grad_tol=1e-4).run()

The positive initial inverse-Hessian model divides reference-Fock-basis
directions by twice the absolute orbital-energy gap times a diagonal
occupation, with occupations floored at 0.1 and curvature floored at 0.1
Hartree. It handles a noncanonical starting MO basis by diagonalizing the
reference RHF Fock matrix. This is a heuristic preconditioner; active-space
two-electron curvature and CI response are learned only approximately
through secants. The complete step still obeys the macro trust radius and
is checked against a fresh CI energy. Exact CASCI and finite-bond DMRG use
their respective physical blocks; fixed-weight state averaging uses weighted
RDMs. No superlinear or quadratic convergence guarantee applies, especially
with approximate CI solves or nonsmooth root changes.

``optimizer`` and the inner tolerance/iteration budget apply only to
``orbital_update="fixed_rdm"``, which remains the default.
``optimizer_history`` controls the retained secant count in both updates.
``optimizer_max_step_norm`` controls the relaxed tangent step, with a
default bound of 0.25. The relaxed update has no frozen-RDM inner loop.
Its default macro rejection budget is 20, and its minimum trust radius is
``1e-8``, allowing late backtracking of an inaccurate secant step. The
fixed-RDM defaults remain eight rejections and a ``1e-4`` minimum radius.
Explicit user values override these defaults.

In the Fe(CO)5 def2-SVP CAS(8,8) CD ``1e-8`` test from the RHF frontier
guess, the preconditioned CI-relaxed update converged in 73 accepted macros
to ``-1825.529432066209`` Hartree with physical gradient ``9.25e-5``.
An independent check of the returned CI/orbital pair confirmed both energy
and gradient. The controlled original-method comparison was bounded at 20
macros, so these data do not establish a full-run macroiteration speedup.
Separately, ``physical_inner=True`` gave gradient ``0.154`` after eight
macros versus ``0.570`` for the original update, with lower energy; that
restricted-inner comparison is a bounded progress test, not convergence.

All modes reject rank-deficient orbital extrapolations before polar
normalization. The macro trust radius bounds the complete step. An
energy-increasing CI trial clears DIIS and retries the ordinary update
before shrinking that radius. ``macro_diagnostics`` includes ``diis_used``,
``diis_space``, ``diis_residual``, and, for extrapolated steps,
``diis_max_weight``, alongside energy and physical-gradient norms.

This is an adaptation of the fixed-RDM CO workflow of J. Zhang, S. Hu and
B. Gu, *Constrained Optimization Algorithms for Orbital Optimization in
Quantum Chemistry* (2026), https://arxiv.org/abs/2606.17761, and the Pulay
least-squares acceleration of P. Pulay, *Improved SCF convergence
acceleration*, J. Comput. Chem. **3**, 556–560 (1982),
https://doi.org/10.1002/jcc.540030413. ``optimizer="ISD"`` implements the paper's
implicit skew-gradient solve, polar projection, nonmonotone Armijo search
and alternating BB steps (Eqs. 16, 33--34), with practical iteration and
step bounds. It uses the complete Stiefel space and rejects
``physical_inner=True``. Other inner optimizers and optional transported
DIIS residuals are adaptations. No coupled CI-response microsteps are
included. Neither quadratic convergence nor a general reduction in
macroiterations is guaranteed; compare both energy and physical-gradient
convergence for the system of interest.

``optimizer="ISD", isd_gap_floor=0.1`` enables positive orbital-gap
preconditioning. In the reference-Fock eigenbasis, the skew generator is
divided elementwise by ``max(2*abs(e_i-e_j), isd_gap_floor)``. Symmetric
positive weights preserve skew-Hermiticity and a descent direction. The
implicit solve and polar projection remain; alternating BB steps use
scaled search-gradient differences, while the stopping test uses the
unscaled gradient. The gap floor retains degenerate and active-active
directions. ``isd_gap_floor=None`` selects plain ISD.

This is a heuristic curvature adaptation inspired by B. Shustin and
H. Avron, *Riemannian optimization with a preconditioning scheme on the
generalized Stiefel manifold* (2023), https://arxiv.org/abs/1902.01635.
It is not their metric construction or an exact orbital Hessian and
inherits no convergence-rate guarantee. The Fe(CO)5 runner enables a
0.1 Hartree floor by default; ``--isd-gap-floor 0`` selects plain ISD.

Active Space Selection
----------------------

By default, PyQED places the active block after the inactive core orbitals. You
can provide explicit zero-based MO indices:

.. code-block:: python

   mc = CASSCF(mf, ncas=4, nelecas=4).run(
       active_orbitals=[2, 3, 4, 5],
   )

Use active orbital selection when:

* the chemically important orbitals are not contiguous,
* the active space changes character along a scan,
* you need to match a PySCF or external active-space reference,
* you want to preserve a manually localized or reordered active block.

The helper checks that the number of active orbital indices equals ``ncas`` and
that there are no duplicates.

Atomic Valence Active Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~

AVAS constructs an active space from chemically meaningful atomic-orbital
labels using native PyQED overlap and cross-basis integrals. Its interface
follows PyQED's ``avas.run`` syntax and accepts a converged PyQED
mean-field object:

.. code-block:: python

   from pyqed.qchem import CASSCF, avas

   ncas, nelecas, mo_avas = avas.run(mf, ["Fe 3d", "N 2p"])
   mc = CASSCF(mf, ncas=ncas, nelecas=nelecas).run(mo_coeff=mo_avas)

The default ``minao="minao"`` reference uses PyQED's bundled minimal ANO-R0
basis. The primary molecule and AVAS use native integrals and do not require
PySCF.

First-Order CASSCF
------------------

``CASSCF`` is the lightweight native orbital optimizer. It:

* solves the active-space CI problem with the native CASCI solver,
* builds spin-traced active-space RDMs,
* forms a generalized Fock matrix,
* computes the nonredundant orbital gradient,
* updates orbitals with diagonal preconditioning and line search.

This path is useful for small and medium active spaces, quick scans, and
testing dense/factorized integral code.

Second-Order CASSCF
-------------------

``SecondOrderCASSCF`` adds a stronger orbital-optimization path with
microiterations:

.. code-block:: python

   from pyqed.qchem import SecondOrderCASSCF

   mc = SecondOrderCASSCF(
       mf,
       ncas=4,
       nelecas=4,
       coupling="full",
       max_micro_cycle=8,
   ).run()

The second-order implementation supports several coupling modes:

* ``coupling="full"`` uses the production full coupling path by default.
* ``coupling="qn"`` uses a quasi-Newton-like orbital path.
* ``coupling="simultaneous"`` performs joint CI-orbital microiterations.
* ``coupling="simultaneous_reduced"`` or ``"simultaneous_partial"`` uses a
  reduced simultaneous coupling.

For routine work, start with ``coupling="full"``. Use simultaneous coupling for
experiments where CI and orbital variables must relax together in the
microiteration.

State Averaging
---------------

State-averaged CASSCF optimizes orbitals for a weighted average of multiple
CASCI roots:

.. code-block:: python

   weights = [0.5, 0.5]
   mc = CASSCF(mf, ncas=4, nelecas=4)
   mc.state_average(weights).run(nstates=2)

State averaging is useful near avoided crossings or when multiple electronic
states must share a consistent orbital basis.

Factorized Integrals
--------------------

CASSCF can use factorized AO ERIs from the RHF reference:

.. code-block:: python

   mol.build(eri="factors")
   mf = mol.RHF().run()
   mc = SecondOrderCASSCF(mf, ncas=4, nelecas=4).run()

In factorized mode, the CASSCF code avoids constructing dense transformed MO
ERI tensors when the active-space contraction can be performed directly with
pair factors:

.. math::

   (pq|rs) \approx \sum_L L^L_{pq} L^L_{rs}.

This is the recommended path for larger basis sets and larger active spaces.

Convergence Controls
--------------------

Important options:

* ``max_cycle`` controls macroiterations.
* ``max_micro_cycle`` controls second-order microiterations.
* ``conv_tol`` controls energy convergence.
* ``conv_tol_grad`` controls strict orbital-gradient convergence.
* ``conv_tol_grad_relaxed`` allows convergence when the energy is stable and
  the gradient is small enough for practical scans.
* ``conv_tol_step`` controls orbital-step convergence.

For production calculations, prefer a slightly larger ``max_cycle`` and monitor
both energy and gradient norms. If the active space changes character along a
scan, use explicit ``active_orbitals`` or orbital-overlap analysis.

QN vs Simultaneous Microiterations
----------------------------------

The quasi-Newton-style path treats the CI response and orbital response in a
more decoupled way. It is usually faster and often robust enough.

The simultaneous path updates CI and orbital variables together inside the
microiteration. It is closer to a fully coupled second-order formulation, but
it is more expensive and more sensitive to trust-region/acceptance settings.

Use this practical rule:

* use ``full`` or ``qn`` for normal calculations,
* use ``simultaneous`` when testing paper-faithful coupled CI-orbital behavior,
* fall back to ``full`` if simultaneous microiterations become too slow or
  reject too many steps.

Examples
--------

Relevant examples:

* ``examples/qchem/casscf.py``
* ``examples/qchem/sa_casscf_factor.py``
* ``examples/qchem/casscf_factor_vs_dense.py``
* ``examples/qchem/mcscf/secondorder_casscf.py``
* ``examples/qchem/benchmark_second_order_casscf.py``
* ``examples/qchem/lif_casscf_scan.py``

Related Pages
-------------

* :doc:`../qchem`
* :doc:`../backends`
* :doc:`../hf_analysis`
* :doc:`../qchem_architecture`
