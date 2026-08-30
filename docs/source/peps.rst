Projected Entangled-Pair States
===============================

``pyqed.peps`` provides finite rectangular projected entangled-pair states
(PEPS) with open boundaries.  Each tensor has canonical index order
``(physical, up, right, down, left)`` and Hamiltonian sites are numbered in
row-major order.

State construction
------------------

Use canonical :class:`pyqed.lattice.Site` objects for the physical spaces::

   from pyqed.lattice import SpinHalfSite
   from pyqed.peps import PEPS

   sites = tuple(SpinHalfSite() for _ in range(4))
   psi = PEPS.random(sites, shape=(2, 2), D=2, seed=7)

``PEPS.product_state`` constructs a bond-one state from basis indices or local
vectors.  ``to_dense`` exactly contracts the physical wavefunction and is a
small-system reference helper.

Contraction
-----------

Norms, overlaps, local products, and structured Hamiltonian expectations
support exact, boundary-MPS, and CTMRG backends::

   norm_exact = psi.norm_squared(method="exact")
   norm_boundary, info = psi.norm_squared(
       method="boundary",
       max_bond=64,
       rtol=1e-10,
       return_info=True,
   )
   norm_ctmrg, ctm = psi.norm_squared(
       method="ctmrg",
       max_bond=64,
       return_info=True,
   )

``method="exact"`` contracts the entire double-layer network and has
exponential cost in lattice width. ``method="boundary"`` absorbs rows into a
boundary MPS and SVD-compresses it. ``max_bond`` bounds the boundary rank;
``rtol`` and ``atol`` remove small singular directions. Setting
``max_bond=None, rtol=0, atol=0`` makes the boundary path exact up to roundoff,
while still retaining row-by-row contraction order. The returned diagnostics
contain retained row ranks and discarded SVD weight.

Finite CTMRG
------------

``method="ctmrg"`` performs four directional corner-transfer
renormalizations. It grows the environment rank geometrically to ``max_bond``
and returns northwest, northeast, southeast, and southwest corner spectra,
directional edge data, a convergence residual, and the spread between the four
directional estimates::

   environment = psi.ctmrg(chi=64)
   print(environment.corners, environment.residual)

For a finite lattice this is a directional boundary/corner CTMRG, not the
fixed-point CTMRG of an infinite translationally invariant iPEPS. With
sufficient ``chi`` and zero truncation tolerance it reproduces the exact
finite-network contraction up to roundoff.

The latest environment is retained as a warm start. A subsequent CTMRG call
begins at its retained ``chi`` instead of repeating geometric rank growth.
``environment.warm_started`` and the per-iteration history expose this choice.

Hamiltonians and observables
----------------------------

The PEPS consumes the same structured :class:`pyqed.tn.Hamiltonian` used by
MPS and LETTA::

   from pyqed.tn import Hamiltonian

   H = Hamiltonian(sites)
   H.add_product(0.25, (0, "X"), (1, "X"))
   energy = psi.expectation(
       H,
       method="boundary",
       max_bond=64,
       workers=4,
   )

Arbitrary finite-support ``LocalTerm`` kernels are operator-Schmidt factored
and evaluated without materializing the full many-body Hamiltonian.
``local_expectation`` accepts a mapping from row-major site indices or grid
coordinates to local matrices.

Hamiltonian expectations build the identity boundary environment once. Each
operator product then starts from the cached prefix immediately before its
support instead of recontracting the complete lattice. This preserves the
previous row-by-row truncation path and therefore gives the same energy at the
same ``max_bond`` and tolerances. CTMRG similarly reuses all four directional
environments. The returned diagnostics report ``environment_reused=True`` and
``environment_builds=1``.

Channels entering at the same cut are grouped into one batched
identity/termwise frontier. Shape-compatible boundary SVDs are evaluated as a
batch while every channel retains its own truncation decision. Diagnostics
include ``batched_frontier=True`` and the total ``frontier_channels``.

The factor contractions are independent after that shared build, so
``workers`` can evaluate them concurrently without changing summation order or
the result. The default is one worker; small matrices and memory-bound large
boundaries can be slower with excessive threading, so benchmark representative
``workers=2`` and ``workers=4`` cases. Worker pools persist across expectation
and evolution calls, avoiding repeated thread startup.

Identity and operator-inserted double layers are cached per site. Replacing a
tensor through PEPS algorithms invalidates only the affected sites. After an
external in-place tensor edit, call ``psi.invalidate_cache()``; direct array
replacement is detected automatically. Energy diagnostics include
``layer_cache_hits`` and ``layer_cache_misses``.

Optimization
------------

The optimizer provides correctness-first one-site and two-site variational
sweeps::

   optimizer = psi.optimize(H, sweeps=4, tol=1e-9)
   optimizer = psi.optimize(H, update="two-site", max_D=4, sweeps=4)
   print(optimizer.energy, optimizer.history)

For an active tensor ``a``, the default one-site path contracts the surrounding
double-layer network with a hole and constructs exact ``H_eff`` and ``N_eff``
matrices. The norm metric is whitened and exactly null gauge directions are
removed. This avoids materializing the exponentially large physical-state
vector, although exact environment contraction remains exponential in lattice
width. Set ``environment="dense"`` only for small reference comparisons.

A one-site update retains all virtual dimensions. A two-site update optimizes
a joined nearest-neighbor block and SVD-splits it, so its shared rank can grow
up to ``max_D``. Truncation error and retained rank are recorded for every
update. The present two-site path uses the dense reference environment and is
therefore protected by ``max_dense_dimension``. Accepted updates cannot
increase the exactly contracted energy beyond roundoff.

``max_local_size`` guards the local generalized eigenproblem. The boundary-MPS
backend permits larger approximate expectation calculations, but an
approximate boundary environment is not silently substituted into the
variational optimizer.

U(1)-blocked PEPS
-----------------

``U1PEPS`` stores only charge-allowed blocks. It uses the oriented flux rule

.. math::

   q_p + q_u + q_r - q_d - q_l = Q_A

for every site tensor. Neighboring bond-sector tables must match. Canonical
``Site.charges`` provide the physical sectors::

   from pyqed.peps import U1PEPS

   charges = {((0, 0), (0, 1)): (-1, 1)}
   psi_u1 = U1PEPS.random(
       sites[:2],
       shape=(1, 2),
       bond_charges=charges,
       target_charges=(0, 0),
       seed=7,
   )
   energy = psi_u1.expectation(H)
   print(psi_u1.block_count, psi_u1.storage_fraction)

Norms and observables contract only compatible double-layer sector paths.
``to_dense_peps`` and ``U1PEPS.from_dense`` provide validation and migration
helpers; forbidden dense entries are rejected unless projection is explicitly
requested.

The default U(1) contractor is a native charge-block frontier. It merges all
row histories with the same vertical sector tuple instead of enumerating whole
lattice sector assignments::

   norm, info = psi_u1.norm_squared(return_info=True)
   reference = psi_u1.norm_squared(method="enumerate")
   directional = psi_u1.norm_squared(method="ctmrg")

``method="enumerate"`` retains the former DFS reference. The frontier is exact
when ``max_frontiers=None, rtol=0, atol=0``. Setting ``max_frontiers`` keeps the
largest sector blocks at each row and reports discarded weight. U(1) CTMRG
evaluates the block frontier in four directions and reports their spread.

Time evolution
--------------

``PEPSEvolution`` supports nearest-neighbor real- and imaginary-time Trotter
evolution. Each two-site gate is applied to a joined pair and SVD-split with
``max_D`` and ``cutoff`` controls::

   dynamics = psi.evolve(H, 1.0, step=0.02, max_D=4)
   cooling = psi.evolve(
       H,
       2.0,
       step=0.02,
       max_D=4,
       imaginary=True,
       measure_every=10,
       workers=4,
   )

The driver stores ``time`` or ``beta``, energy, norm changes, retained ranks,
and discarded weights in ``history``. For ``U1PEPS``, gates are split by
charge-sector SVD, ranks are selected globally across sectors, and new bond
charges can appear without leaving block-sparse storage. A charge-breaking
gate is rejected rather than projected silently.

``measure_every`` controls the expensive Hamiltonian contraction while always
measuring the final step. Skipped records contain ``energy=None`` and
``measured=False``. The norm before a step is reused from the preceding
normalized step, avoiding a redundant contraction.

Hot double-layer, row-absorption, boundary-joining, and gate operations use
fixed ``tensordot`` or matrix-multiplication layouts. Exact finite-grid
contraction expressions and local operator factorizations are cached by shape
and kernel, so repeated calculations do not rerun path searches or TT-SVDs.

Example
-------

Run the open-boundary Heisenberg example with::

   PYTHONPATH=. python examples/peps/heisenberg_2d.py --rows 2 --cols 2 --D 2
   PYTHONPATH=. python examples/peps/u1_exchange_dynamics.py
