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
     "pyqed_hamiltonian": [[...], [...]],
     "pyqed_state_charges": [[...], [...]],
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

Staged frame/ambient optimization and a diagonal latent ``A`` are available
as experimental ``fit_y`` options. Simultaneous training with a full Hermitian
``A`` remains the default because it currently gives the best SO2 held-out
Hamiltonian accuracy.

Both model classes support checkpoints. Native MACE calculations require the
optional dependency group ``pyqed[mace]``.
