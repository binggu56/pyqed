Discrete Variable Representation
================================

Discrete variable representations (DVRs) describe a wavefunction by its
values on a grid.  They are especially convenient when the potential is local:
the potential-energy operator is diagonal, while PyQED supplies the
representation-specific kinetic-energy matrix.

When to use a DVR
-----------------

Use a DVR for low-dimensional nuclear or model-coordinate problems when you
want to:

* solve a one- or few-dimensional Schrödinger equation on a finite domain;
* apply a potential directly as values evaluated on a grid;
* inspect wavefunctions in coordinate space; or
* construct a grid Hamiltonian for later propagation or coupling to another
  model.

The sine DVR used below is a good default for a bounded, non-periodic
coordinate whose wavefunction is negligible at both ends of the interval.  A
periodic coordinate, a singular radial problem, or a large multidimensional
calculation may require a different representation.

Minimal example: harmonic oscillator
------------------------------------

This example constructs

.. math::

   H = -\frac{1}{2m}\frac{d^2}{dx^2} + \frac{x^2}{2}

in atomic units.  ``SineDVR`` provides the interior grid ``dvr.x`` and the
kinetic-energy matrix ``dvr.t()``; a local potential is added as a diagonal
matrix.

.. literalinclude:: ../../examples/dvr/sine_harmonic_oscillator.py
   :language: python
   :linenos:

Expected result
~~~~~~~~~~~~~~~

.. code-block:: text

   [0.5 1.5 2.5 3.5]

These are the first four analytic harmonic-oscillator energies,
:math:`E_n=n+1/2`.  Agreement here checks both the box and grid for these
low-lying states; it does not by itself establish convergence for higher
states or a different potential.

Important parameters
--------------------

``xmin``, ``xmax``
   Endpoints of the finite box.  Sine-basis functions vanish at the
   endpoints, and the ``npts`` coordinates lie strictly inside the box.

``npts``
   Number of sine basis functions and interior DVR points.  Increasing it
   improves spatial resolution but makes the dense matrices larger.

``mass``
   Mass used by the kinetic-energy operator.  It defaults to ``1``.  The mass,
   coordinate, potential, and requested energies must all use one consistent
   unit system.

``dvr.x``
   The coordinate values at which a local potential is evaluated.

``dvr.t()``
   The analytic sine-DVR kinetic-energy matrix.  Assemble a one-dimensional
   local Hamiltonian as ``dvr.t() + np.diag(V(dvr.x))``.

How the sine DVR works
----------------------

On a box of length :math:`L=x_{\max}-x_{\min}`, the finite-basis
representation consists of sine functions that satisfy zero boundary
conditions.  PyQED transforms this basis to the equally spaced interior
collocation points

.. math::

   x_\alpha = x_{\min} + \frac{\alpha L}{N+1},
   \qquad \alpha=1,\ldots,N.

The potential is diagonal at these points.  The kinetic operator is not
diagonal in the DVR, but its matrix elements are known analytically.  The
result is a straightforward dense eigenvalue problem for small grids and a
convenient Hamiltonian representation for time propagation.

Convergence and limitations
---------------------------

Converge a reported result with respect to both the box and ``npts``:

1. Enlarge ``xmin`` and ``xmax`` until the states of interest are negligible
   near both boundaries.
2. Increase ``npts`` until energies and observables stop changing at the
   required precision.
3. Check normalization and boundary behavior of the corresponding
   eigenvectors, not only their eigenvalues.

The standard sine-DVR matrices are dense, requiring :math:`O(N^2)` storage;
full dense diagonalization costs :math:`O(N^3)`.  Tensor-product grids grow
exponentially with the number of coordinates.  Discontinuous potentials,
continuum states, periodic coordinates, and coordinate singularities also
need special care or a more suitable representation.

Other DVR implementations
-------------------------

The :mod:`pyqed.dvr` namespace also contains sinc, finite-element,
Gauss--Hermite, simultaneous-diagonalization, and multidimensional DVR tools.
Their grids, boundary conditions, and numerical tradeoffs differ, so do not
change the DVR family without repeating the convergence checks.

API, source, and related material
---------------------------------

* API namespace: :mod:`pyqed.dvr`; ``SineDVR`` is implemented in
  :mod:`pyqed.dvr.dvr_1d`.
* `SineDVR source (PyQED 0.2.0)
  <https://github.com/binggu56/pyqed/blob/v0.2.0/pyqed/dvr/dvr_1d.py>`__
* `Runnable harmonic-oscillator example
  <https://github.com/binggu56/pyqed/blob/main/examples/dvr/sine_harmonic_oscillator.py>`__
* `Sine-DVR/FEDVR comparison example (PyQED 0.2.0)
  <https://github.com/binggu56/pyqed/blob/v0.2.0/examples/dvr/fedvr_vs_sine_quartic.py>`__
* :doc:`tutorials` for the recommended learning path
* :doc:`examples` for additional grid calculations
* :doc:`pyqed.namd` and :doc:`geometric_quantum_dynamics` for dynamics built on
  coordinate-space representations
