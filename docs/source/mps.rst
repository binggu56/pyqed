Matrix Product States
=====================

Matrix product states (MPS) are compact tensor-network representations for
one-dimensional quantum many-body states. PyQED uses MPS ideas for lattice
models, vibrational problems, quantum chemistry DMRG, and
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

For large Abelian bond dimensions, ``DMRG(..., opt="3s")`` selects the
strictly one-site DMRG3S/AMEn path.  Its local Davidson vector is a rank-3
U(1)-blocked tensor rather than a rank-4 two-site tensor.  Partial residuals
``L W A`` and ``A W F`` enrich the next bond without constructing a two-site
effective Hamiltonian::

   solver = DMRG(
       H,
       D=512,
       init_guess=psi,
       opt="3s",
       target_qn=target,
       enrichment=1e-4,
   ).run()

A labelled ``target_qn`` automatically enables Abelian symmetry and supplies
the symmetry labels; a separate ``sym_mgr`` argument is unnecessary.

``opt="1site"`` is an alias for this implementation.  It currently supports
one Abelian target state and an explicit block-sparse MPO; use ``opt="2site"``
for dense, state-averaged, SU(2), or complementary-operator calculations.

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
and total spin. PyQED contains Abelian and non-Abelian/SU(2) paths.

For SU(2)-adapted quantum chemistry, tensors store reduced multiplet data
rather than all spin components. This requires explicit Clebsch-Gordan and
fusion-tree bookkeeping, but can substantially reduce the number of states
needed for spin-adapted calculations.

The quantum-chemistry SU(2) driver is a strict compiled backend. It uses fully
reduced spatial sites and a C++-owned normal/complementary half-sweep; it raises
if the compiled route is unavailable instead of switching to Python bond
callbacks. The generic Python non-Abelian contractions remain available for
models, LETTA, and reference validation.

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

* ``pyqed.tn`` contains the canonical dense ``MPS``, ``MPO``,
  ``Hamiltonian``, and analytical operator-string compiler.
* ``pyqed.mps`` contains explicit MPS algorithms such as DMRG, TEBD, TDVP,
  and uniform or continuous MPS drivers.
* ``pyqed.operator_mpo`` contains the specialized ``ModelMPO`` compiler used
  by vibronic and grid Hamiltonian models.
* ``pyqed.mps.nonabelian`` contains reduced SU(2)/non-Abelian tensors,
  ``AutoMPO``, and sweep components.
* ``pyqed.qchem.dmrg`` contains quantum-chemistry DMRG and DMRG-SCF-facing
  code.
* ``pyqed.dmrg`` contains older/simple DMRG examples and prototypes.

Examples
--------

Useful example entry points:

* ``examples/mps/model_mpo_fermion.py``
* ``examples/mps/model_mpo_boson.py``
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
