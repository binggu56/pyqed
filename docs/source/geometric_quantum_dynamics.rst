Geometric Quantum Dynamics
==========================

Geometric quantum dynamics describes nuclear motion on electronic-state
manifolds where phases, overlaps, derivative couplings, and coordinate metrics
matter. PyQED contains tools for locally diabatic representations, overlap
matrices between electronic states, curvilinear DVR propagation, Gaussian
wavepacket dynamics, and Born-Oppenheimer Hamiltonian derivatives.

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

PyQED's LDR-related code lives mainly in ``pyqed.ldr``:

* ``pyqed.ldr.ldr``: locally diabatic representation wavepacket dynamics.
* ``pyqed.ldr.qd``: quasi-diabatization from overlap matrices.
* ``pyqed.ldr.curvilinear`` and ``pyqed.ldr.curvilinear_2d``: curvilinear DVR
  propagation and overlap-modified kinetic propagation.
* ``pyqed.ldr.gwp``: Gaussian wavepacket basis and overlap utilities.
* ``pyqed.ldr.coarse_grained``: coarse-grained overlap-based dynamics.

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
