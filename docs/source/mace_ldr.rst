Atomistic MACE fields for LDR
==============================

PyQED separates transferable atomistic learning from molecule-specific tensor
compression.

``MACEStateModel`` consumes variable-size Cartesian molecular records and
predicts a Hermitian multistate Hamiltonian together with state-specific atomic
charges. Predictions are conditioned on total molecular charge, multiplicity,
an electronic-structure fidelity label, and a state-manifold label. Atomic
charges are corrected by a differentiable projection so that every state obeys
exact charge conservation.

QCSchema-compatible records can be supplied with two PyQED extension fields in
``extras``:

.. code-block:: json

   {
     "pyqed_hamiltonian": [[0.0, 0.1], [0.1, 1.0]],
     "pyqed_state_charges": [[0.0, 0.0], [0.1, -0.1]],
     "pyqed_manifold": "singlet-S0S1"
   }

QCSchema geometry is interpreted in bohr and converted to the Angstrom
convention used by ASE and MACE. The method and basis form the default fidelity
label. The explicit manifold label prevents, for example, an ``S0/S1`` matrix
from being mixed with a different pair of states that happens to have the same
method and basis. A trained transferable model can expose its predictions through
``AbInitioFit`` using ``model.abinitio_fit(...)``.

``MACE`` is the molecule-specific adapter. It learns aligned energy/link or
endpoint-feature fields on one nuclear-coordinate chart, then distills them to
``FunctionalTT`` for ``TTLDR``. Cartesian unit conventions must be declared
with ``geometry_units``. ``chart_features=False`` is the transferable
atomistic-only mode; enabling chart features is an explicitly coordinate-aided
interpolation baseline.

Endpoint-frame training uses ``feature_objective="links-only"`` by default,
so it trains the frame only through ``Y_i^dagger Y_j`` and does not regress a
synchronized gauge. Endpoint targets and neural predictions are retracted to
the Stiefel manifold, making ``Y^dagger Y=I`` an exact architectural constraint
rather than a penalty. The alternative ``feature_objective="subspace"`` compares
``Y Y^dagger`` projectors, which do not change under a local right-unitary
rotation.  The legacy ``"fixed"`` objective is retained only for diagnostics.
Finite-rank FunctionalTT distillation is an entrywise approximation and reports
its residual isometry defect explicitly.

The energy head predicts an ambient Hermitian field ``A(R)`` and evaluates
``H(R) = Y(R)^dagger A(R) Y(R)``.  Therefore ``Y -> Y G`` gives
``H -> G^dagger H G``, while endpoint links transform as
``L_ij -> G_i^dagger L_ij G_j``.
``MACE.predict_covariant(coordinates, gauges)`` returns both fields in a
requested set of local gauges; without ``gauges`` it returns the learned
canonical chart.

For conical-intersection Hamiltonians, ``fit_y`` accepts
``energy_objective="trace-traceless"``.  It normalizes the scalar mean surface
and the traceless branching-space field independently before combining their
losses.  This still regresses the complete Procrustes-gauged local Hamiltonian;
it does not fit or order the individual adiabatic eigenvalues.  The balanced
objective is useful when the mean PES has a much larger amplitude than the
symmetry-covariant splitting field.

``refine_hamiltonian`` continues training only the direct MACE Hamiltonian
head.  The atomistic encoder and endpoint field ``Y`` are frozen, so the raw
links cannot drift during an H-only refinement.  With ``finite_group`` the
head is evaluated through the same Reynolds projection as ordinary MACE
inference, making ``H(C_g q) = D_g H(q) D_g^dagger`` exact.  This is still a
MACE model; no polynomial PES surrogate is substituted before FunctionalTT
distillation.

Staged frame/ambient optimization and a diagonal latent ``A`` are available
as experimental ``fit_y`` options. Simultaneous training with a full Hermitian
``A`` remains the default because it currently gives the best SO2 held-out
Hamiltonian accuracy.

Both model classes support checkpoints. Native MACE calculations require the
optional dependency group ``pyqed[mace]``.

The encoder follows the MACE construction of I. Batatia *et al.*, *Advances in
Neural Information Processing Systems* **35**, 11423--11436 (2022),
`arXiv:2206.07697 <https://arxiv.org/abs/2206.07697>`_.  The PyQED LDR model is
an adaptation, not an exact reproduction: it uses invariant pooled atomistic
features, coordinate-chart matrix heads, Procrustes-gauged Hamiltonian and raw
overlap targets, and no force loss.  Accuracy guarantees for a trained MACE
interatomic potential therefore do not transfer to these electronic fields.

Production native build
-----------------------

For a native molecular electronic driver, ``AbInitioFit.build()`` owns the
complete production route:

.. code-block:: python

   fit = AbInitioFit(
       mc,
       coord=Coord(to_cartesian=geometry, bounds=bounds),
       states=(1, 2),
   ).build()

The build detects molecular permutation symmetry, decomposes the coordinate
action into irrep blocks, infers the selected-state action on separate
Procrustes-gauged orbits, and rejects a leaking state manifold.  It trains a
MACE ensemble on quotient-domain points, calibrates ensemble spread on one
continuous-coordinate set, and applies Hamiltonian/link error and uncertainty
coverage gates on a disjoint set.  Only an accepted neural model is distilled
to FunctionalTT; the TT energy and endpoint-field errors are then gated again.
``fit.acceptance`` and ``fit.validation`` contain the thresholds and measured
errors.  ``fit.save(path)`` stores both the distilled fields and ``mace.pt``.

The default thresholds are deliberately strict and may reject a calculation.
Changing them through ``fit_options`` changes the qualification criterion; it
does not make a rejected model physically accurate.  ``model="ftt"`` selects
the older direct adaptive FunctionalTT path explicitly, while ``model="mace"``
requires the atomistic MACE path.

After building a TNLDR, a small direct-product grid can gate the complete
surrogate and MPO path:

.. code-block:: python

   direct = fit.direct_product(dynamics_grid, keo=nuclear_keo)
   tnldr.validate(direct, initial_tensor, dt=dt, steps=steps)

This exact finite-space check compares the Hamiltonian, populations, norm, and
wavefunction fidelity.  It is intended as a qualification test on tractable
grids, not as the production propagator.

Finite-group quotient sampling
------------------------------

``AbInitioFit`` detects molecular atom permutations and their induced action on
the supplied coordinate chart.  It then calibrates the selected-state action
from Procrustes-gauged electronic Hamiltonians.  A finite group reduces
scattered designs to one representative per coordinate orbit; the aligned
coordinate and state representations make the MACE Hamiltonian and endpoint
field exactly equivariant:

.. code-block:: python

   from pyqed.ldr import AbInitioFit, Coord

   coord = Coord(to_cartesian=geometry, bounds=bounds)
   sampler = AbInitioFit(mc, coord=coord, states=(1, 2))

   representatives = sampler.reduce_coordinates(candidate_coordinates)
   pair_coordinates, pairs = sampler.reduce_pairs(coordinates, pairs)
   finite_group = sampler.mace_group(feature_rank=16)

The detected results are available directly as ``sampler.group``,
``sampler.coord_repr``, and ``sampler.state_repr``.  Detection belongs to the
fit rather than ``Coord`` because it requires both ``mc.mol`` and the
coordinate embedding.  Pass ``symmetry=False`` to disable detection or a
low-level group action through ``symmetry=...`` when an unusual chart needs an
explicit override.

Pair reduction applies one common group operation to both endpoints.  The link
target remains the raw overlap and is transported covariantly; it is never
replaced by a Procrustes/polar unitary.  Coordinate-only group actions also do
not alias electronic database records, because doing so without transforming
the full electronic frame would corrupt subsequent overlaps.  Symmetries such
as ``PhenolReflectionSymmetry`` that implement full record transport may safely
share a canonical database record among orbit images.

Ethylene conical-intersection benchmark
----------------------------------------

``examples/namd/ethylene_ci_2d_tnldr.py`` provides a two-dimensional
twisted--pyramidalized ethylene benchmark.  Its production grid uses a periodic
Fourier DVR for torsion and a finite sine DVR for pyramidalization.  The default
electronic calculation is equally weighted SA(2)-CASSCF(2,2).  The
electronic database defaults to
``~/Library/CloudStorage/OneDrive-西湖大学/data/pyqed/ethylene_ci_periodic_2d``;
the driver rejects a database path inside the PyQED repository.

Prepare and inspect the ab initio database without training MACE:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/ethylene_ci_2d_tnldr.py --prepare-only

Build the direct periodic benchmark used for dynamics:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/ethylene_ci_2d_tnldr.py --direct-only

Then run matched direct-LDR and raw-link TNLDR wavepacket dynamics through the
crossing:

.. code-block:: bash

   PYTHONPATH=. python examples/namd/ethylene_ci_2d_dynamics.py

The dynamics driver launches the same upper-adiabatic Gaussian in the direct
and tensor-network representations of the identical raw-link Hamiltonian.  It
checks TDVP error against dense propagation and repeats TDVP2 with half the time
step.  No absorber is used.  This remains a frozen-coordinate 2D
illustration rather than a full-dimensional ethylene photodynamics model.

The optional endpoint-frame MACE route is experimental on this full periodic
chart.  A single globally periodic endpoint field cannot in general represent
nontrivial torsional holonomy; use a multipatch field with an explicit seam
transition before treating such a fit as a production surrogate.

The twisted--pyramidalized MRCI MECI geometry used in *Machine Learning Seams
of Conical Intersection: A Characteristic Polynomial Approach*, J. Phys. Chem.
Lett. (2023), DOI
`10.1021/acs.jpclett.3c01649 <https://doi.org/10.1021/acs.jpclett.3c01649>`_.
is the source template.  Because the smaller electronic model has a displaced
seam, the coordinate origin applies an additional pyramidalization of about
-0.725 radian to place the default SA(2)-CASSCF(2,2)/6-31G* calculation at its
restricted-chart crossing.  This implementation is an adaptation rather than
a reproduction: its finite torsion and pyramidalization deformations are not
optimized branching vectors, SA-CASSCF omits MRCI dynamic correlation, and all
remaining nuclear coordinates are frozen.  ``--electronic-method casci`` is
intended only for inexpensive workflow smoke tests.
