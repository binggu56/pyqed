Projected-Entangled-Pair States
===============================

``pyqed.peps`` provides finite open-boundary PEPS on rectangular lattices.
Site tensors use the fixed ordering ``(left, right, up, down, physical)``.
Simple-update states use gamma/lambda storage: site tensors contain the gamma
tensors and singular values are stored once per nearest-neighbor bond.

The initial implementation includes:

* dense and Abelian block-sparse state containers,
* product states,
* exact contractions for small reference lattices,
* approximate boundary-MPS norms and local expectation values,
* nearest-neighbor real- or imaginary-time gates, and
* dense simple-update bond truncation.

Boundary contraction treats every double-layer PEPS row as an MPO acting on
an MPS.  ``max_bond=None`` leaves this boundary MPS untruncated; finite values
set its environment bond dimension.

.. code-block:: python

   import numpy as np

   from pyqed.peps import PEPS, simple_update_bond, two_site_gate

   zero = np.array([1.0, 0.0])
   psi = PEPS.product_state([zero] * 4, (2, 2))

   sx = np.array([[0, 1], [1, 0]])
   h_bond = np.kron(sx, sx)
   gate = two_site_gate(h_bond, 0.01, imaginary=True)
   simple_update_bond(
       psi,
       (0, 0),
       (0, 1),
       gate,
       max_bond=8,
   )

   norm = psi.norm_squared(method="boundary", max_bond=32)

Abelian site tensors store only nonzero charge blocks.  Exact and boundary-MPS
contractions accept them by materializing individual site tensors.  A native
charge-preserving block SVD for simple update is not yet provided; attempting
an Abelian simple update raises ``NotImplementedError`` instead of silently
discarding symmetry.
