Theory Overview
===============

PyQED collects methods for molecular quantum dynamics, spectroscopy,
light-matter interaction, open quantum systems, and quantum chemistry. The
common starting point is a Hamiltonian model and a representation for the
state, density matrix, or response function.

Hamiltonian Models
------------------

Most workflows start from a Hamiltonian split into a reference part and one or
more interactions:

.. math::

   H(t) = H_0 + V(t).

For molecular problems, ``H_0`` may be an electronic Hamiltonian, a vibronic
Hamiltonian, a grid Hamiltonian, or an effective model Hamiltonian. The
time-dependent perturbation ``V(t)`` often represents a laser pulse, a cavity
mode, or a coupling to an external field.

The time-dependent Schrodinger equation is

.. math::

   i \frac{\partial}{\partial t} |\psi(t)\rangle = H(t)|\psi(t)\rangle,

and the corresponding density-matrix equation is

.. math::

   \frac{d\rho}{dt} = -i [H(t), \rho].

Atomic units are used in most quantum chemistry and molecular dynamics code
paths unless a specific example states otherwise.

Quantum Chemistry
-----------------

The nonrelativistic electronic Hamiltonian in a molecular orbital basis is

.. math::

   H =
   \sum_{pq} h_{pq} a_p^\dagger a_q
   + \frac{1}{2}\sum_{pqrs} (pq|rs)
     a_p^\dagger a_r^\dagger a_s a_q
   + E_\mathrm{nuc}.

Here ``h`` contains one-electron kinetic and nuclear-attraction integrals,
``(pq|rs)`` are electron-repulsion integrals, and ``E_nuc`` is the nuclear
repulsion energy.

Hartree-Fock approximates the wavefunction by a single determinant and solves
self-consistent orbital equations:

.. math::

   F C = S C \varepsilon.

Post-HF methods add correlation on top of this reference. PyQED includes CI,
MP2, CASCI, and CASSCF-related workflows. In active-space methods, the orbital
space is partitioned into inactive, active, and external orbitals. CASCI solves
the active-space CI problem for fixed orbitals, while CASSCF optimizes both the
CI coefficients and orbital rotations.

For larger calculations, electron-repulsion integrals may be represented by
factors rather than a dense four-index tensor:

.. math::

   (pq|rs) \approx \sum_L L^L_{pq} L^L_{rs}.

This reduces memory pressure and enables direct factorized contractions in
RHF, CASCI, and CASSCF workflows.

Related pages:

* :doc:`qchem`
* :doc:`backends`
* :doc:`hf_analysis`
* :doc:`mp2_comp2`
* :doc:`guide/guide_qchem_mcscf`

Discrete Variable Representation
--------------------------------

Discrete variable representation methods represent wavefunctions on a grid
while keeping derivative operators as matrices. A typical one-dimensional grid
Hamiltonian is

.. math::

   H_{ij} = T_{ij} + V(x_i)\delta_{ij}.

The potential is diagonal in the DVR grid, while the kinetic-energy matrix
depends on the selected DVR basis. DVR methods are useful for wavepacket
dynamics, model potentials, and grid-based quantum chemistry prototypes.

For tensor-product grids, multidimensional Hamiltonians are assembled from
one-dimensional kinetic terms and a multidimensional potential:

.. math::

   H = \sum_\alpha I_1 \otimes \cdots \otimes T_\alpha
       \otimes \cdots \otimes I_d + V(q_1,\ldots,q_d).

Related page:

* :doc:`dvr`

Nonlinear Spectroscopy
----------------------

Spectroscopy observables are response functions of dipole or current
operators. Linear absorption is determined by a two-point dipole correlation
function:

.. math::

   C(t) = \langle \mu(t)\mu(0)\rangle.

The frequency-domain spectrum is obtained from a Fourier transform, with
appropriate damping or windowing:

.. math::

   I(\omega) \propto \omega\, \mathrm{Im}
   \int_0^\infty dt\, e^{i\omega t} C(t).

Nonlinear spectroscopy uses higher-order response functions. For example, a
third-order signal depends on nested commutators of dipole operators evaluated
at multiple time delays. In practice, PyQED supports both sum-over-states and
time-domain propagation viewpoints.

Related page:

* :doc:`guide/guide_spectroscopy`

Open Quantum Systems
--------------------

Open-system dynamics describe a system coupled to an environment. A common
Markovian model is the Lindblad master equation:

.. math::

   \frac{d\rho}{dt}
   = -i[H,\rho]
   + \sum_k \left(
       L_k \rho L_k^\dagger
       - \frac{1}{2}\{L_k^\dagger L_k,\rho\}
     \right).

For structured baths, the environment is often characterized by a spectral
density:

.. math::

   J(\omega) = \sum_k |g_k|^2 \delta(\omega - \omega_k).

The bath correlation function determines memory effects and dissipative rates.
HEOM and related methods approximate the bath correlation function as a sum of
exponentials and propagate auxiliary density operators.

Related pages:

* :doc:`guide/guide_open_dynamics`
* :doc:`heom`

Floquet Theory
--------------

For a periodic Hamiltonian,

.. math::

   H(t + T) = H(t),

solutions can be written in Floquet form:

.. math::

   |\psi_\alpha(t)\rangle =
   e^{-i\epsilon_\alpha t} |\phi_\alpha(t)\rangle,
   \qquad
   |\phi_\alpha(t+T)\rangle = |\phi_\alpha(t)\rangle.

The quasienergies ``epsilon`` are eigenvalues of the Floquet operator in an
extended time-periodic Hilbert space. This representation is useful for driven
two-level systems, periodically driven lattices, and light-dressed matter.

Related page:

* :doc:`pyqed.floquet`

Light-Matter Coupling
---------------------

Light-matter models couple electronic, vibrational, or model-system degrees of
freedom to quantized or classical fields. A minimal cavity-QED Hamiltonian has
the form

.. math::

   H = H_\mathrm{mol}
   + \omega_c a^\dagger a
   + g\,\mu\,(a^\dagger + a),

where ``omega_c`` is the cavity frequency, ``g`` is the coupling strength, and
``mu`` is the molecular dipole operator. Depending on gauge and approximation,
additional self-polarization terms may be required.

Related page:

* :doc:`pyqed.polariton`

Nonadiabatic and Ehrenfest Dynamics
-----------------------------------

Nonadiabatic dynamics couple nuclear motion to multiple electronic states. In
an adiabatic basis, nuclear motion feels state-dependent energies and derivative
couplings:

.. math::

   d_{IJ}(R) = \langle \Phi_I(R) | \nabla_R \Phi_J(R) \rangle.

Ehrenfest dynamics propagates classical nuclei on the mean field generated by a
coherent electronic state:

.. math::

   M_A \ddot{R}_A = -\nabla_{R_A}
   \langle \Psi_e(t;R) | H_e(R) | \Psi_e(t;R) \rangle.

In a locally diabatic overlap formulation, short-time electronic propagation
can use state overlaps between neighboring geometries instead of explicit
nonadiabatic coupling vectors. This is useful when analytic NACs are
unavailable or expensive.

Related page:

* :doc:`pyqed.namd`
* :doc:`tddft_ehrenfest`
* :doc:`geometric_quantum_dynamics`

Method Selection
----------------

Use these rough guidelines:

* Use RHF as the starting point for closed-shell mean-field calculations.
* Use MP2 for inexpensive dynamic correlation near a single-reference state.
* Use CASCI when active orbitals are fixed and static correlation matters.
* Use CASSCF when active orbitals need to relax with the CI wavefunction.
* Use TDDFT or TDHF for linear-response excited states and spectra.
* Use DVR methods for low-dimensional grid dynamics.
* Use MPS/DMRG when an active space or lattice model is too large for exact
  diagonalization but has manageable entanglement.
* Use HEOM or master equations when environmental dissipation is essential.
* Use Floquet theory when the Hamiltonian is periodic in time.

Related tensor-network page:

* :doc:`mps`
