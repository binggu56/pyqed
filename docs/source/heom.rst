HEOM and structured baths
=========================

The base installation provides the conventional HEOM solver:

.. code-block:: python

   from pyqed.HEOM.heom import HEOMSolver

The dissipaton-equation-of-motion (DEOM) and structured-bath helpers use
Numba, SymPy, and tqdm. Install their declared extra before importing them:

.. code-block:: bash

   python -m pip install "pyqed[heom]"

Then import the DEOM solver from the same documented package namespace:

.. code-block:: python

   from pyqed.HEOM.deom import DEOMSolver

These modules are research workflows. Preserve the PyQED version, bath
decomposition, hierarchy cutoff, integration settings, and convergence checks
with any reported result.

.. automodule:: pyqed.HEOM
   :members:
   :undoc-members:
   :show-inheritance:
