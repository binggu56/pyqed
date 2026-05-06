Discrete Variable Representation
================================

PyQED includes several discrete variable representation (DVR) utilities for
grid-based quantum dynamics and model Hamiltonians. DVR methods represent
wavefunctions on quadrature or collocation grids while keeping derivative and
kinetic-energy operators as compact matrices.

Available Modules
-----------------

The DVR code lives under ``pyqed.dvr``:

* ``pyqed.dvr.dvr_1d`` provides one-dimensional DVR utilities.
* ``pyqed.dvr.dvr_2d`` provides multidimensional tensor-product DVR utilities.
* ``pyqed.dvr.sddvr`` implements simultaneous-diagonalization DVR helpers.
* ``pyqed.dvr.gwp_fbr`` contains Gaussian-wavepacket finite-basis tools for
  SD-DVR.
* ``pyqed.dvr.gauss_hermite`` contains legacy Gauss-Hermite DVR routines.

Typical Workflow
----------------

A DVR calculation usually follows this pattern:

1. Choose a coordinate domain and grid size.
2. Build the kinetic-energy operator for the chosen DVR basis.
3. Evaluate the potential on the DVR grid.
4. Assemble the Hamiltonian ``H = T + V``.
5. Diagonalize or propagate the wavefunction.

Example
-------

The one-dimensional utilities can be used to assemble a simple grid
Hamiltonian. The exact class/function names depend on the DVR module selected,
but the calculation structure is:

.. code-block:: python

   import numpy as np

   from pyqed.dvr.dvr_1d import kinetic

   x = np.linspace(-8.0, 8.0, 256)
   mass = 1.0

   T = kinetic(x, mass=mass, dvr="sinc")
   V = np.diag(0.5 * x**2)
   H = T + V

   energies, states = np.linalg.eigh(H)
   print(energies[:5])

Notes
-----

This page is intentionally written as a static guide rather than an autodoc API
page. Some legacy DVR modules import optional dependencies or run example code
at import time, which is not suitable for warning-free Read the Docs builds.
