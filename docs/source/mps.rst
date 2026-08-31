Matrix Product States
=====================

Matrix product states (MPS) are compact tensor-network representations for
one-dimensional quantum many-body states. PyQED uses MPS ideas for lattice
models, vibrational problems, quantum chemistry DMRG prototypes, and
time-dependent simulations.

MPS Ansatz
----------

For a chain of ``L`` sites with local basis states ``sigma_i``, an MPS writes
the wavefunction as

.. math::

   |\Psi\rangle =
   \sum_{\sigma_1\cdots\sigma_L}
   A^{\sigma_1}_1 A^{\sigma_2}_2 \cdots A^{\sigma_L}_L
   |\sigma_1\cdots\sigma_L\rangle.

Each ``A_i`` is a rank-3 tensor with one physical index and two virtual bond
indices. The maximum virtual dimension ``D`` controls the expressive power:
larger ``D`` captures more entanglement but increases cost.

The Schmidt decomposition across a bond is

.. math::

   |\Psi\rangle =
   \sum_{\alpha=1}^{\chi}
   s_\alpha
   |\alpha_L\rangle |\alpha_R\rangle,

where the number of significant singular values determines the required bond
dimension. Low-entanglement states are therefore efficient in MPS form.

Canonical Forms
---------------

MPS algorithms rely on canonical gauges. A left-canonical tensor satisfies

.. math::

   \sum_\sigma (A^\sigma)^\dagger A^\sigma = I,

while a right-canonical tensor satisfies

.. math::

   \sum_\sigma A^\sigma (A^\sigma)^\dagger = I.

Mixed-canonical form places an orthogonality center on one site or bond. This
is the preferred representation for local optimization, expectation values,
and stable time evolution.

Matrix Product Operators
------------------------

Operators can be represented as matrix product operators (MPOs):

.. math::

   \hat{O} =
   \sum_{\sigma,\sigma'}
   W^{\sigma_1\sigma'_1}_1
   W^{\sigma_2\sigma'_2}_2
   \cdots
   W^{\sigma_L\sigma'_L}_L
   |\sigma_1\cdots\sigma_L\rangle
   \langle\sigma'_1\cdots\sigma'_L|.

For lattice models, compact MPOs are often available analytically. For quantum
chemistry, the Hamiltonian contains long-range two-electron terms and requires
careful MPO construction or complementary-operator factorizations.

DMRG
----

The density matrix renormalization group (DMRG) variationally minimizes

.. math::

   E = \frac{\langle\Psi|H|\Psi\rangle}
            {\langle\Psi|\Psi\rangle}

within the MPS manifold. In a one-site or two-site sweep, all tensors except a
local block are held fixed, producing an effective eigenvalue problem:

.. math::

   H_\mathrm{eff} x = E x.

Sweeping repeatedly through the chain relaxes the MPS toward the ground state
or targeted low-lying states.

Shared-memory parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~

Finite DMRG accepts ``n_threads`` for shared-memory execution.  The sweep and
the Davidson iteration sequence remain serial, while independent output blocks
of the native dense effective-Hamiltonian matvec and scalar environment-update
kernels use OpenMP.  The same setting is forwarded to Numba's parallel Abelian
contraction kernels.  For example:

.. code-block:: python

   dmrg = DMRG(H, D=256, init_guess=psi0, n_threads=8)
   dmrg.run()

``dmrg.threading_info`` records which native and Numba runtimes were available
and the resolved thread counts.  Native OpenMP is optional: if the extension
was built without it, dense DMRG retains the serial native/BLAS path.  Keep the
BLAS thread count at one when using outer DMRG threads to avoid nested thread
oversubscription.

Quantum-chemistry SU(2) DMRG accepts the same spelling, ``n_threads``::

   solver = qcdmrg.run(symmetry="su2", D=256, n_threads=8)

Here OpenMP acts only on reduced-sector local-operator rows and independent
reduced complementary-operator execution batches.  A dependency-wave
scheduler groups executions with disjoint outputs and runs them inside one
persistent OpenMP region.  The default output-affinity executor writes
disjoint reduced-sector blocks directly.  If output conflicts leave that
executor with fewer usable threads, a memory-gated executor instead computes
reusable first-stage products once, schedules whole fused contractions into
worker-private outputs, and combines those outputs with a tree reduction.
It retains already-combined GEMMs and is selected only when it exposes more
parallelism than output affinity.  The private replicas are limited to 128 MiB
per bond by default; ``PYQED_SU2_PRIVATE_OUTPUT_BYTES`` changes that limit and
``0`` disables this executor.  The implementation never expands the SU(2)
state into determinants.
``solver.diagnostics["threading"]`` and each sweep-history entry report the
compiled OpenMP availability, selected thread count, and executed parallel
work.  Set ``PYQED_MPS_OPENMP=0`` while building to request a serial extension,
or ``PYQED_MPS_OPENMP=1`` to require OpenMP.  On macOS, automatic discovery
prefers Homebrew ``libomp`` over the active Conda environment; set
``PYQED_OPENMP_PREFIX`` to select another runtime explicitly.  The selected
runtime is linked directly by both native DMRG extensions so the runtime-built
dense Davidson module cannot preload a different ``libomp.dylib`` before the
SU(2) kernel.

Infinite-system DMRG
--------------------

For translationally invariant infinite chains, ``pyqed.mps`` also exposes
``iDMRG`` and the convenience wrapper ``idmrg_nearest_neighbor``.  This iDMRG
path is deliberately separate from the finite
quantum-chemistry sweep stack: it starts from a dense nearest-neighbor bond
Hamiltonian, factors it into local channels, grows persistent left/right
renormalized block Hamiltonians and boundary operators, solves the two-site
infinite-system superblock, and truncates through the center Schmidt spectrum.
``iDMRG.run()`` populates the solver object with ``state``, ``history``,
``energy_density``, ``center_bond_energy``, and block data.  The reported
``energy_density`` is the growth/incremental iDMRG estimate; the finite
superblock energy per site and candidate uniform-state energy are kept in
``metadata`` for diagnostics.  Plain infinite growth does not always produce a
consistent repeating tensor, so ``state`` is populated only when the candidate
``UniformMPS`` energy agrees with the growth estimate within the configured
``state_energy_tol``.

Quantum Chemistry DMRG
----------------------

In quantum chemistry, each spatial orbital or spin orbital is mapped to a site.
The electronic Hamiltonian is

.. math::

   H =
   \sum_{pq} h_{pq} a_p^\dagger a_q
   + \frac{1}{2}\sum_{pqrs} (pq|rs)
     a_p^\dagger a_r^\dagger a_s a_q.

DMRG is useful when the active space is too large for full CI but the
entanglement structure is still moderate. It can be used as a CASCI solver or
as the active-space solver inside DMRG-SCF/CASSCF workflows.

Symmetries
----------

Symmetry-adapted MPS implementations reduce cost by block-sparsifying tensors.
Common symmetries include particle number, spin projection, point group labels,
and total spin. PyQED contains both Abelian and prototype non-Abelian/SU(2)
development paths.

For SU(2)-adapted quantum chemistry, tensors store reduced multiplet data
rather than all spin components. This requires explicit Clebsch-Gordan and
fusion-tree bookkeeping, but can substantially reduce the number of states
needed for spin-adapted calculations.

Cross-geometry SU(2) overlaps
-----------------------------

Two completed fully reduced SU(2) DMRG calculations can be compared across
different molecular geometries with ``dmrg_bra.overlap(dmrg_ket)`` or
``dmrg_bra.overlap_biorthogonal(dmrg_ket, backend="su2")``.  PyQED first
builds the cross-geometry AO overlap, eliminates the inactive-core coupling,
and biorthogonalizes the active orbital spaces.  It then factors each
nonunitary active-orbital map into diagonal scalings and adjacent two-orbital
Gaussian gates.  Every gate is applied directly to the reduced charge x SU(2)
channel blocks, including the exact intermediate-spin recoupling, before an
identity-MPO contraction of the transformed MPSs.

The biorthogonalization follows P.-A. Malmqvist, *Int. J. Quantum Chem.* **30**,
479--494 (1986), `doi:10.1002/qua.560300404
<https://doi.org/10.1002/qua.560300404>`_.  The use of nonunitary
transformations for nonorthogonal MPS state interaction follows S. Knecht,
S. Keller, J. Autschbach, and M. Reiher, *J. Chem. Theory Comput.* **12**,
5881--5894 (2016), `doi:10.1021/acs.jctc.6b00889
<https://doi.org/10.1021/acs.jctc.6b00889>`_.  The adjacent reduced-sector
circuit is a PyQED adaptation; it is not a line-by-line reproduction of either
reference implementation.

The practical defaults are ``cutoff=1e-10`` and ``max_bond="auto"``.  If the
input reduced MPS bond dimension is :math:`D`, the adaptive cap is

.. math::

   D_{\mathrm{overlap}} =
   \max\left[D,\min\left(8192,\max(256,16D)\right)\right].

This never compresses below the input dimension but bounds transformation-
induced growth in normal use.  The factor and limits were selected from
reduced-SU(2) truncation tests: a chemically structured H10/CAS(10,10) test
required approximately :math:`16D` to recover the untruncated overlap, whereas
:math:`4D` and :math:`8D` produced material errors.  The result remains a
controlled MPS approximation.  Set ``cutoff=0`` and ``max_bond=None``
explicitly to discard no singular values and obtain a result exact up to
floating-point roundoff.
With ``return_info=True``, each transformed state reports its resolved bond
cap, peak reduced bond dimension, sum of the per-gate relative discarded
weights, maximum relative discarded weight at one gate, and number of gates
that truncated data.  The sum is a convergence diagnostic, not a rigorous
bound on the final overlap error.

The active path does not recover determinant amplitudes, create a
spin-component MPS, or allocate a ``4**ncas`` state.  Its practical cost
instead follows the bond dimensions generated by the orbital circuit.  An
arbitrary untruncated orbital transformation can produce exponential
entanglement and therefore exponential bond growth in the worst case; sector
preservation does not imply a fixed polynomial bound.  The current SU(2) route
requires matching numbers of active and inactive orbitals and a nonsingular
effective active overlap.

Time-Dependent MPS
------------------

Time evolution applies

.. math::

   |\Psi(t+\Delta t)\rangle
   \approx e^{-iH\Delta t}|\Psi(t)\rangle.

MPS time evolution can be implemented with TEBD, TDVP-like updates, Krylov
local propagation, or problem-specific MPO exponentials. PyQED includes
time-dependent MPS examples for model and quantum chemistry workflows.

Uniform MPS
-----------

For translationally invariant infinite chains, PyQED provides a lightweight
``UniformMPS`` class in ``pyqed.mps``.  It stores a one-site uMPS tensor
``A[s, left, right]`` or a finite unit-cell stack
``A[i, s, left, right]`` and supports transfer fixed points, local density
matrices, nearest-neighbor expectation values, and correlation-length
estimates.  One-site tensors additionally support canonical gauges,
mixed-canonical data, entanglement spectra, and ``vumps_nearest_neighbor`` for
dense tangent-space VUMPS updates.  ``optimize_nearest_neighbor`` and
``optimize_nearest_neighbor_unit_cell`` provide compact direct variational
fallbacks for small model checks.  ``iDMRG`` provides a
growth-based infinite-system DMRG route that returns ``UniformMPS`` data when
the retained center layout is compatible.  Optimizers return the optimized
``UniformMPS`` directly and store metadata such as ``energy``, ``success``, and
``nfev`` on that state.  The optional
``examples/mps/uniform_mps_vs_tenpy.py`` script compares the Heisenberg energy
density against TeNPy infinite VUMPS when TeNPy is installed.

Uniform LETTA
-------------

``pyqed.letta`` also provides ``UniformLETTA`` for terminal uniform LETTA in
the thermodynamic limit.  Its tensor convention follows the finite LETTA code:
``A[left, s, t, right]`` stores a nearest-neighbor pair tensor whose physical
legs are shared between neighboring factors.  ``UniformLETTA`` contracts its
own transfer matrix, computes one- and two-site density matrices, and provides
``optimize_nearest_neighbor`` to vary the LETTA pair tensor entries directly.
``UniformLETTA.from_uniform_mps`` embeds any same-unit-cell ``UniformMPS`` by
choosing the LETTA tensor independent of the shared right physical leg.

Package Map
-----------

The MPS-related code is split across several namespaces:

* ``pyqed.mps`` contains general MPS, MPO, TEBD, DMRG, symmetry, and AutoMPO
  utilities, plus ``UniformMPS`` for one-site uniform/infinite MPS work.
* ``pyqed.mps.autompo`` contains automatic MPO construction helpers.
* ``pyqed.mps.nonabelian`` contains prototype SU(2)/non-Abelian tensor and DMRG
  components.
* ``pyqed.qchem.dmrg`` contains quantum-chemistry DMRG and DMRG-SCF-facing
  code.
* ``pyqed.dmrg`` contains older/simple DMRG examples and prototypes.

Examples
--------

Useful example entry points:

* ``examples/mps/autompo.py``
* ``examples/mps/autompo_boson.py``
* ``examples/mps/hydrogen_chain.py``
* ``examples/mps/nonabelian_hubbard_chain_benchmark.py``
* ``examples/mps/nonabelian_hubbard_solver_scaling.py``
* ``examples/qchem/dmrgscf.py``
* ``examples/qchem/mps_domain_wall_q_tdmps.py``
* ``examples/qchem/mps_three_electrons_q_dmrg.py``
* ``examples/qchem/tddmrg_h2_threeway_compare.py``

Related Pages
-------------

* :doc:`qchem`
* :doc:`examples`
* :doc:`nonabelian_dmrg_design`
