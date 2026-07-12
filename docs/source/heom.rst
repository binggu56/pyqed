Hierarchical equations of motion
================================

The hierarchical equations of motion (HEOM) propagate a reduced density
matrix together with auxiliary density operators (ADOs) that retain bath
memory.  Compared with a time-local Lindblad model, HEOM is useful when a
structured environment or non-Markovian response matters.

When to use HEOM
----------------

Use the current PyQED HEOM solver for compact model studies with:

* a finite-dimensional system Hamiltonian;
* one system--bath coupling operator;
* a Drude--Lorentz bath represented by one exponential correlation term; and
* dynamics that can be converged with a modest hierarchy depth.

For a weak, rapidly relaxing bath, a Lindblad or Redfield calculation may be
less expensive.  Baths that require several exponential terms, low-temperature
corrections, or production-scale hierarchy filtering are beyond the current
compact solver and should not be approximated without validation.

Minimal example: spin--boson dynamics
-------------------------------------

The following calculation starts in the second diabatic state, couples
:math:`\sigma_z` to a Drude--Lorentz bath, and records
:math:`\langle\sigma_z\rangle` for 100 time steps.

.. literalinclude:: ../../examples/heom_compact.py
   :language: python
   :linenos:

Expected result
~~~~~~~~~~~~~~~

.. code-block:: text

   Final <sigma_z>: -0.96907844

``expect`` is the observable history returned by ``run``, not a density-matrix
trajectory.  Its shape is ``(len(e_ops), nt)``; ``expect[0]`` is therefore the
sampled :math:`\langle\sigma_z\rangle` trace.

Important parameters
--------------------

``H``
   System Hamiltonian as a square array.

``rho0``
   Initial reduced density matrix.  Check that it is Hermitian, positive, and
   has unit trace before propagation.

``c_ops``
   System--bath coupling operators.  The current HEOM path uses the first
   operator, ``c_ops[0]``.

``e_ops``
   Operators whose expectation values are returned at every step.  Supply at
   least one observable.

``dt``, ``nt``
   Integration step and number of steps.  The reported time interval is
   ``nt * dt`` in the same inverse-energy/time convention used by ``H``.

``temperature``
   Bath thermal energy :math:`k_B T` in the solver's unit convention.  The
   current ``run`` path uses the value directly; it does not convert kelvin.

``cutoff``
   Drude cutoff :math:`\gamma`, the inverse bath-memory time.  Larger values
   describe a faster, more nearly Markovian bath.

``reorganization``
   Reorganization energy :math:`\lambda`, which controls the bath-coupling
   strength.

``nado``
   Number of hierarchy slots allocated, including the physical density matrix
   and the zero terminator.  Increase it until the observable is insensitive
   to further increases in ``nado``.

Bath model and hierarchy
------------------------

The implemented Drude--Lorentz spectral density is

.. math::

   J(\omega) = \frac{2\lambda\gamma\omega}
                     {\omega^2+\gamma^2}.

PyQED represents its bath correlation with one decaying term,

.. math::

   C(t) \simeq D_0 e^{-\gamma t}, \qquad
   D_0 = \lambda\gamma
   \left[\coth\!\left(\frac{\gamma}{2T}\right)-i\right],

where :math:`T` denotes the supplied thermal energy.  The physical reduced
density matrix is the zeroth ADO.  Higher ADOs encode successive orders of
bath memory and are coupled to their neighboring tiers.  PyQED closes the
finite hierarchy by setting the next, unrepresented ADO to zero and advances
the equations on the uniform ``dt`` grid.

Convergence and limitations
---------------------------

A single trajectory is not a convergence test.  For a reported calculation:

1. Repeat it with a smaller ``dt``.
2. Increase ``nado`` until all observables of interest are stable.
3. Extend ``nt`` to cover the full relaxation or correlation time.
4. Monitor Hermiticity, trace, and positivity of the reduced state in a
   validation calculation.
5. Record the PyQED version, unit convention, bath parameters, hierarchy
   depth, and integration settings.

All Hamiltonian, bath, temperature, and time parameters must be expressed in
one consistent unit system.  In particular, passing ``temperature=600`` does
not by itself mean 600 K; the short example uses the solver's native numerical
parameter convention to reproduce the stated regression value.

The current implementation uses one coupling operator and one exponential
bath term, a simple zero terminator, fixed-step propagation, and stores only
expectation values from ``run``.  The ratio :math:`\gamma/T` is printed as a
diagnostic; the solver warns when its thermal approximation may be unreliable.
Strong coupling, slow baths, low temperature, or long propagation commonly
require a deeper hierarchy and a more complete bath decomposition.

DEOM and structured-bath workflows
----------------------------------

The dissipaton-equation-of-motion (DEOM) and structured-bath research modules
use additional dependencies.  Install the declared extra before importing
them:

.. code-block:: bash

   python -m pip install "pyqed[heom]"

These workflows are separate from the compact ``pyqed.oqs.HEOMSolver`` entry
point above.  Inspect their examples and preserve every decomposition and
truncation parameter when publishing results.

API, source, and related material
---------------------------------

* Solver entry point: ``pyqed.oqs.HEOMSolver`` (an alias of
  ``pyqed.oqs.HEOM``).
* `HEOM solver source (PyQED 0.2.0)
  <https://github.com/binggu56/pyqed/blob/v0.2.0/pyqed/oqs.py>`__
* `Runnable compact HEOM example
  <https://github.com/binggu56/pyqed/blob/main/examples/heom_compact.py>`__
* `Longer historical HEOM example (PyQED 0.2.0)
  <https://github.com/binggu56/pyqed/blob/v0.2.0/examples/heom.py>`__
* :doc:`guide/guide_open_dynamics` for open-system models and bath notation
* :doc:`tutorials` for the recommended learning path
* :doc:`examples` for Lindblad, Redfield, DEOM, and spectroscopy examples
