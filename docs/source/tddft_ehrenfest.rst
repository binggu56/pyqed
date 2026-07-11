TDDFT and Ehrenfest Dynamics
============================

PyQED contains native linear-response TDDFT/TDA tools and an Ehrenfest dynamics
driver that can use TDDFT state data. The dynamics layer supports both
NAC-based propagation and an overlap-based locally diabatic formulation.

Linear-Response TDDFT
---------------------

The native TDDFT implementation lives in ``pyqed.qchem.tddft``. It builds the
restricted singlet A/B response matrices for RHF or native RKS references:

.. math::

   \begin{pmatrix}
   A & B \\
   -B & -A
   \end{pmatrix}
   \begin{pmatrix}
   X \\
   Y
   \end{pmatrix}
   =
   \omega
   \begin{pmatrix}
   X \\
   Y
   \end{pmatrix}.

TDA solves only the Hermitian A-block problem:

.. math::

   A X = \omega X.

Basic usage:

.. code-block:: python

   mf = mol.RHF().run()

   td = mf.TDDFT().run(nstates=3)
   print(td.e)

   tda = mf.TDA().run(nstates=3)
   print(tda.e)

Current native support is intentionally limited: RHF/TDHF-like response and
LDA-family RKS TDDFT are the stable native paths. More advanced XC kernels may
require a backend comparison or a PySCF-backed workflow.

Gradients
---------

``pyqed.qchem.tddft.Gradients`` provides a PySCF-backed analytic
excited-state-gradient adapter for TDDFT/TDA objects when PySCF is installed:

.. code-block:: python

   td = mf.TDDFT().run(nstates=3)
   grad = td.nuc_grad_method(backend="pyscf").kernel(state=1)

The native linear-response solver is still used for excitation energies. The
gradient backend mirrors the calculation in PySCF because fully native TDDFT
Z-vector response machinery is not yet implemented.

TDDFTDriver
-----------

``TDDFTDriver`` is a thin state-data adapter used by ``TDDFTEhrenfest``. It
provides:

* ``evaluate(coords) -> (energies, grads, nac)``
* ``as_scanner()``
* ``point_data(coords)``
* ``state_overlap(coords_ref, coords_other)``
* ``normal_modes()``

Example:

.. code-block:: python

   from pyqed.namd.mf import TDDFTDriver

   driver = TDDFTDriver(
       mol,
       nstates=3,
       xc="lda,vwn",
       build_driver="builtin",
       nac_method="none",
   )

   scanner = driver.as_scanner()
   energies, grads, nac = scanner(mol.atom_coords())

Ehrenfest Equations
-------------------

Ehrenfest dynamics propagates classical nuclei and a coherent electronic state:

.. math::

   i\dot{c}_I =
   E_I c_I
   - i\sum_J \dot{R}\cdot d_{IJ} c_J,

.. math::

   M_A \ddot{R}_A =
   -\sum_{IJ} c_I^* c_J
   \langle \Phi_I | \nabla_A H_e | \Phi_J \rangle.

In the adiabatic NAC representation, derivative couplings are

.. math::

   d_{IJ}(R) = \langle \Phi_I(R) | \nabla_R \Phi_J(R) \rangle.

Overlap-Based Local Diabatic Propagation
----------------------------------------

For the overlap-based path, PyQED computes the state overlap between adjacent
geometries:

.. math::

   S_{IJ}(t,t+\Delta t)
   =
   \langle \Phi_I(R_t) | \Phi_J(R_{t+\Delta t}) \rangle.

The overlap is unitarized to define a local diabatic transformation. Electronic
propagation and force construction are then performed in this locally diabatic
frame, avoiding explicit derivative coupling vectors for the propagation step.

Usage:

.. code-block:: python

   from pyqed.namd.mf import TDDFTEhrenfest, TDDFTDriver

   driver = TDDFTDriver(mol, nstates=3, xc="lda,vwn", nac_method="overlap_fd")
   dyn = TDDFTEhrenfest(mol, ntraj=20, nstates=3, nac_driver=driver)

   dyn.sample(init_state=1, distribution="thermal_wigner", temperature=300.0)
   dyn.run(dt=0.1, nt=100, electronic_representation="overlap")

Use ``electronic_representation="overlap"`` for the locally diabatic overlap
formulation. The default non-overlap path uses the wrapped scanner and NAC
array returned by the driver.

Wigner Sampling
---------------

``TDDFTEhrenfest.sample()`` initializes nuclear positions and momenta from
normal modes. The default distribution is thermal Wigner sampling:

.. math::

   \sigma_q^2 =
   \frac{1}{2\omega}\coth\left(\frac{\omega}{2 k_B T}\right),
   \qquad
   \sigma_p^2 =
   \frac{\omega}{2}\coth\left(\frac{\omega}{2 k_B T}\right).

The implementation also accepts explicit ``q_var`` and ``p_var`` if the user
wants to override the frequency-based defaults.

Choosing NAC or Overlap
-----------------------

* Use NAC propagation when reliable analytic or finite-difference NACs are
  available and you want the traditional adiabatic formulation.
* Use overlap propagation when state overlaps are more robust or cheaper than
  explicit NACs.
* Use ``nac_method="none"`` for purely adiabatic state energies/gradients with
  zero derivative coupling.
* Use ``nac_method="overlap_fd"`` when finite-difference overlaps should be
  converted into approximate NACs.

Related Pages
-------------

* :doc:`pyqed.namd`
* :doc:`qchem`
* :doc:`theory`
* :doc:`geometric_quantum_dynamics`
