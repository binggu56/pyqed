Open quantum dynamics
=====================

Open-system methods describe a subsystem whose environment is not propagated
explicitly.  The central object is the reduced density matrix
:math:`\rho(t)`.  Its equation of motion is written in Liouville space as

.. math::

   \frac{d\rho}{dt} = \mathcal{L}(t)\rho,

where the Liouvillian contains the system Hamiltonian and the selected model
of environmental relaxation and memory.

Choosing a method
-----------------

.. list-table:: Open-system starting points
   :header-rows: 1
   :widths: 19 34 47

   * - Method
     - Appropriate starting regime
     - Main checks
   * - Lindblad
     - Markovian dynamics with known jump operators and non-negative rates
     - Trace preservation, positivity, rate and unit conventions
   * - Redfield
     - Weak system--bath coupling with a specified bath spectrum
     - Born--Markov/secular assumptions and possible positivity loss
   * - Time-convolutionless
     - A time-local perturbative generator for weak coupling
     - Perturbative order, validity time, and generator convergence
   * - HEOM
     - Structured baths or non-Markovian memory represented by exponential
       correlation terms
     - Time step, hierarchy depth, bath decomposition, trace, and positivity

This table is a starting point, not an automatic selector.  Derive the
approximations from the physical time and energy scales of the problem, then
validate a small case independently.

Lindblad master equation
------------------------

For a Hamiltonian :math:`H` and jump operators :math:`L_k`, a standard
Markovian generator is

.. math::

   \frac{d\rho}{dt}
   = -i[H,\rho]
   + \sum_k \gamma_k
     \left(L_k\rho L_k^\dagger
     - \frac{1}{2}\{L_k^\dagger L_k,\rho\}\right).

The rates :math:`\gamma_k` and operators :math:`L_k` define the physical
model; they should not be chosen merely to reproduce a desired decay curve.
Check that the propagated state remains Hermitian, has unit trace, and is
positive within numerical precision.

Redfield and time-convolutionless equations
-------------------------------------------

Redfield theory derives a weak-coupling generator from system operators and
bath correlation functions.  A secular approximation may improve positivity
but can remove physically relevant coherences when transition frequencies are
near-degenerate.  Time-convolutionless approaches instead construct a
time-local perturbative generator.  In both cases, record the spectral
density, temperature convention, perturbative order, and every Markov or
secular approximation.

HEOM and bath memory
--------------------

Hierarchical equations of motion retain environmental memory through
auxiliary density operators.  PyQED's compact current entry point implements
a single-exponential Drude--Lorentz bath for small model studies.  The
:doc:`HEOM guide </heom>` contains a runnable spin--boson example, expected
output, parameter definitions, and explicit limitations.

A common model is

.. math::

   H = \frac{\Delta}{2}\sigma_x + \frac{\epsilon}{2}\sigma_z
     + \sigma_z\sum_k \left(g_k a_k^\dagger + g_k^*a_k\right)
     + \sum_k \omega_k a_k^\dagger a_k,

with bath spectral density

.. math::

   J(\omega) = \sum_k |g_k|^2\delta(\omega-\omega_k).

The compact HEOM solver uses the Drude--Lorentz form

.. math::

   J(\omega) = \frac{2\lambda\gamma\omega}
                    {\omega^2+\gamma^2}.

Here :math:`\lambda` is the reorganization energy and :math:`\gamma^{-1}` is
the bath-memory timescale.  See :doc:`heom </heom>` for the correlation
approximation and solver conventions actually implemented.

Reliable workflow
-----------------

#. Write the system Hamiltonian, initial state, coupling operators, bath
   spectrum, and unit convention explicitly.
#. Check that :math:`\rho(0)` is Hermitian, positive, and normalized.
#. Run the smallest model and monitor trace, Hermiticity, positivity, and any
   conserved quantities.
#. Tighten the time step and each method-specific truncation independently.
#. Compare with an analytic limit or an independent solver where possible.
#. Preserve the version, input, dependencies, and raw trajectory.

Related material
----------------

* :doc:`/heom` -- runnable HEOM calculation and convergence controls
* :doc:`/theory` -- density-matrix and open-system conventions
* :doc:`/tutorials` -- recommended learning paths
* :doc:`/examples` -- compact and research example inventory
* :doc:`/capabilities` -- current maturity and evidence boundaries
