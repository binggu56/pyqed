# Quantum trajectory methods

`jastrow_1d.py` contains the maintained one-dimensional benchmark code:

- `ProjectedJastrow1D` performs deterministic overdamped relaxation on a
  positive Jastrow density manifold. Fixed quantile trajectories carry the
  density, and the residual classical-plus-quantum force is projected onto
  the manifold tangent fields.
- `LegacyPolynomialQTM1D` preserves the older frictional trajectory idea in a
  runnable form, with the score and momentum fields fitted in a polynomial
  basis.
- `exact_quartic_ground_state` supplies a finite-difference reference.
- `ProjectedTwoParticleJastrow1D` adds a genuine pair-Jastrow correlation hole
  for two interacting particles. A deterministic Rosenblatt map transports a
  fixed set of seeded i.i.d. uniform Monte Carlo labels into the correlated
  joint density only at initialization. The equally weighted coordinates are
  then integrated explicitly with the projected tangent velocity and are not
  reset to the inverse-CDF map. Energy, parameter gradients, the tangent
  metric, and force projections are all estimated from the carried particles;
  the Cartesian grid is used only to tabulate inverse conditional CDFs and as
  an independent validation reference. The RMS difference from the exact
  fixed-label transport map is reported as a numerical drift diagnostic.
  Coupled fourth-order Runge--Kutta integration advances both the parameters
  and coordinates; no trajectory retraction or weight update is performed.
  By default, however, the coordinates are the only propagated state:
  $(a,b,c)$ are reconstructed algebraically at every Runge--Kutta stage using
  the Stein identity
  $\langle\mathbf f\cdot\nabla\log\psi\rangle
  =-\langle\nabla\cdot\mathbf f\rangle/2$ with
  $\mathbf f_a=\nabla B_a$. Set `parameter_closure="coupled"` to retain the
  explicitly integrated parameter reference path.
  By default, tangent fields are gradients of symmetry-adapted polynomial
  potentials obtained by minimizing the Monte Carlo weak-Poisson functional.
  The KKT solve simultaneously enforces $JU=I$ for the implicitly
  differentiated Stein reconstruction and the finite-cloud identity
  $\mathcal F=-\nabla_\theta E$. Its dimension is set by the basis size rather
  than the number of particles or configuration-space dimension.
  `tangent_backend="stein"` selects the simpler
  reconstruction-only lift, while `tangent_backend="transport"` retains the
  earlier Rosenblatt-map finite differences as a reference.
  Its quantum force has interchangeable `analytic` and JAX `ad` backends.

The two-particle model has one physical coordinate per particle. Its
configuration space $(x_1,x_2)$ is therefore two-dimensional, even though the
particles move in one physical dimension rather than in a two-dimensional
plane.

Run the quartic-oscillator comparison from the repository root with

```bash
PYTHONPATH=. python examples/qt/quartic_jastrow.py
PYTHONPATH=. python examples/qt/two_particle_jastrow.py
PYTHONPATH=. python examples/qt/three_particle_jastrow.py
PYTHONPATH=. python examples/qt/local_transport_scaling.py
PYTHONPATH=. python examples/qt/direct_score_double_well.py
PYTHONPATH=. python examples/qt/proximal_score_double_well.py
PYTHONPATH=. python examples/qt/difficult_double_well_vmc.py
PYTHONPATH=. python examples/qt/tdvmc_tunneling.py
```

`transport_basis.py` provides the scalable transport representation used by
the two-particle solver: compact, shared one- and two-body radial features,
optional three-body edge features selected by weak-objective improvement, and
particle-count-independent feature dimension. `neural_transport.py` provides
an experimental shared invariant neural scalar potential trained with the
same weak-Poisson objective. Gradients of either invariant scalar
representation are permutation- and rotation-equivariant velocity fields.

The two-particle driver accepts three transport representations:

```python
solver = ProjectedTwoParticleJastrow1D(transport_basis="local")

# Experimental learned invariant features. Train once on the initial cloud;
# every propagation stage recomputes the hard-constrained linear readout.
solver = ProjectedTwoParticleJastrow1D(transport_basis="neural")
x0 = solver.sample_initial(np.log((1.2, 0.5, 0.5)))
theta0, _ = solver.reconstruct_parameters(x0)
solver.train_neural_transport(x0, theta0, steps=150)
solver.run(theta0=theta0)
```

The neural hidden-feature kinetic matrix is whitened and rank-truncated before
the KKT solve. This prevents increasing hidden width from degrading the exact
$JU=I$ and force-gradient constraints through redundant features.

`direct_score_flow.py` is an experimental beyond-Jastrow path. Fixed-weight
particles carry the density and move with the full Cartesian residual force,
$\gamma\dot R=F_{\rm cl}+F_Q$, with no tangent projection, parameter velocity,
momentum, or Langevin force. `score_corrections.py` provides a shared linear
one-/two-/three-body score correction and a smooth invariant neural amplitude
correction. The asymmetric-double-well benchmark is intentionally diagnostic:
although both online closures lower the finite-cloud local-energy estimate,
their residual-force norm and local-energy variance grow. Thus the current
explicit refit-and-move loop is not yet a stable ground-state solver; a
force-consistent proximal update or joint score/transport variational step is
needed before this path should be used for production calculations.

`transport_proximal_flow.py` implements the consistent weak/proximal variant.
Rather than treating a fitted score as an independent vector field, it composes
invertible, permutation-equivariant maps of the initial Jastrow density and
uses the exact change of variables
$\rho_T(T(Z))=\rho_0(Z)/|\det\nabla T|$. The weak kinetic energy is therefore
the energy of a normalized positive state and remains variational. The map is
optimized on fixed training labels; separate fixed audit labels and deterministic
three-dimensional quadrature expose Monte Carlo bias. This route evaluates map
derivatives through second order but never evaluates a pointwise quantum force
or its third amplitude derivatives.

`double_well_vmc.py` supplies a deliberately difficult three-particle VMC
benchmark: a high symmetric double well with soft repulsion.  It separates
occupation-sector trapping from pair-Jastrow ansatz bias by comparing local
Metropolis sampling, an exact-symmetry global reflection move, an explicit
three-body amplitude feature, and Jacobian-consistent composed maps against a
sparse three-dimensional reference calculation.

`tdvmc.py` supplies a real-time complex-Jastrow TDVMC reference and the exact
one-dimensional continuity lift of its density tangent. The tunneling example
compares grid-quadrature TDVMC, fixed-weight continuity-corrected trajectories,
raw Bohmian trajectory quadrature, and split-operator propagation. It reports
both the physical phase current and the corrected sampling current; these are
different whenever the finite Jastrow manifold has a nonzero continuity
residual. The corrected trajectory calculation closes the loop: its complex
Jastrow parameters are estimated and advanced from the moving fixed-weight
quantiles. The raw Bohmian cloud is retained as a passive control because it
does not remain distributed as the evolving finite-Jastrow density.

The `1D/` and `solids/` directories contain the original research scripts and
the parallel solid-helium implementation. They are retained as legacy
references and are not part of the maintained API.
