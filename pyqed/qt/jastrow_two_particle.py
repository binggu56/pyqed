"""Projected trajectories for two interacting particles in one dimension."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import eigsh

from .jastrow_1d import quartic_potential
from .neural_transport import InvariantNeuralTransportPotential
from .transport_basis import SharedRadialTransportBasis


def soft_coulomb(x1, x2, interaction=1.0, softening=0.5):
    """Return a repulsive soft-Coulomb interaction."""
    return interaction / np.sqrt((np.asarray(x1) - np.asarray(x2)) ** 2 + softening**2)


def exact_two_particle_ground_state(
    *,
    anharmonicity=4.0,
    interaction=1.0,
    softening=0.5,
    mass=1.0,
    hbar=1.0,
    xmax=4.5,
    ngrid=121,
):
    """Return the positive two-particle ground state on a square grid."""
    if ngrid < 11 or ngrid % 2 == 0:
        raise ValueError("ngrid must be an odd integer of at least eleven")
    full_grid = np.linspace(-xmax, xmax, ngrid)
    dx = full_grid[1] - full_grid[0]
    x = full_grid[1:-1]
    size = x.size
    kinetic_1d = -(hbar**2 / (2.0 * mass * dx**2)) * diags(
        (np.ones(size - 1), -2.0 * np.ones(size), np.ones(size - 1)),
        (-1, 0, 1),
        format="csr",
    )
    kinetic = kron(kinetic_1d, eye(size), format="csr") + kron(
        eye(size), kinetic_1d, format="csr"
    )
    x1, x2 = np.meshgrid(x, x, indexing="ij")
    potential = (
        quartic_potential(x1, anharmonicity)
        + quartic_potential(x2, anharmonicity)
        + soft_coulomb(x1, x2, interaction, softening)
    )
    hamiltonian = kinetic + diags(potential.ravel(), format="csr")
    energy, state = eigsh(hamiltonian, k=1, which="SA")
    psi = state[:, 0].reshape(size, size)
    psi /= np.sqrt(np.trapezoid(np.trapezoid(psi**2, x, axis=1), x))
    if psi[size // 2, size // 2] < 0.0:
        psi *= -1.0
    return x, psi, float(energy[0])


class ProjectedTwoParticleJastrow1D:
    r"""Overdamped projected dynamics for a correlated positive state.

    The log amplitude is

    $$
    A(x_1,x_2)=-\frac{a}{2}(x_1^2+x_2^2)
      -b[h_s(x_1)+h_s(x_2)]
      -c\exp[-(x_1-x_2)^2/(2\sigma^2)],
    $$

    where $(a,b,c)=\exp(\boldsymbol\theta)$.  The final term is a genuine
    pair Jastrow correlation hole for repulsive particles.
    """

    def __init__(
        self,
        *,
        ntraj=4096,
        seed=7,
        mass=1.0,
        hbar=1.0,
        anharmonicity=4.0,
        interaction=1.0,
        softening=0.5,
        envelope_smoothing=0.35,
        pair_width=0.75,
        friction=1.0,
        xmax=4.5,
        ngrid=181,
        tangent_step=2.0e-4,
        regularization=1.0e-9,
        force_backend="analytic",
        tangent_backend="continuity",
        continuity_order=3,
        transport_basis="local",
        local_transport_basis=None,
        neural_transport_model=None,
    ):
        if ntraj < 16:
            raise ValueError("ntraj must be at least sixteen")
        positive = (mass, hbar, softening, envelope_smoothing, pair_width, friction)
        if any(value <= 0.0 for value in positive):
            raise ValueError("mass, hbar, lengths, and friction must be positive")
        self.ntraj = int(ntraj)
        self.seed = int(seed)
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.anharmonicity = float(anharmonicity)
        self.interaction = float(interaction)
        self.softening = float(softening)
        self.envelope_smoothing = float(envelope_smoothing)
        self.pair_width = float(pair_width)
        self.friction = float(friction)
        self.tangent_step = float(tangent_step)
        self.regularization = float(regularization)
        if force_backend not in {"analytic", "ad"}:
            raise ValueError("force_backend must be 'analytic' or 'ad'")
        self.force_backend = force_backend
        if tangent_backend not in {"continuity", "stein", "transport"}:
            raise ValueError(
                "tangent_backend must be 'continuity', 'stein', or 'transport'"
            )
        self.tangent_backend = tangent_backend
        if int(continuity_order) < 1:
            raise ValueError("continuity_order must be positive")
        self.continuity_order = int(continuity_order)
        if transport_basis not in {"local", "polynomial", "neural"}:
            raise ValueError(
                "transport_basis must be 'local', 'polynomial', or 'neural'"
            )
        self.transport_basis = transport_basis
        if local_transport_basis is None:
            local_transport_basis = SharedRadialTransportBasis()
        self.local_transport_basis = local_transport_basis
        if neural_transport_model is None:
            neural_transport_model = InvariantNeuralTransportPotential(
                3, hidden_width=16, seed=self.seed
            )
        if neural_transport_model.nparameters != 3:
            raise ValueError("the neural transport model must have three outputs")
        self.neural_transport_model = neural_transport_model
        self.physical_dimension = 1
        self.configuration_dimension = 2
        self._ad_point_kernel = None
        self.grid = np.linspace(-xmax, xmax, int(ngrid))
        self.quantiles = np.random.default_rng(self.seed).random((self.ntraj, 2))
        self.weights = np.full(self.ntraj, 1.0 / self.ntraj)

        self.theta = None
        self.initial_x = None
        self.x = None
        self.energy = None
        self.gradient = None
        self.reference_energy = None
        self.reference_gradient = None
        self.transport_drift = None
        self.stein_residual = None
        self.stein_condition = None
        self.parameter_velocity = None
        self.kinematic_error = None
        self.lift_error = None
        self.continuity_residual = None
        self.force_gradient_gap = None
        self.history = None
        self.success = False
        self.message = "not run"

    def _envelope(self, x):
        s = self.envelope_smoothing
        radius = np.sqrt(x**2 + s**2)
        h = (radius**3 - s**3) / 3.0
        dh = x * radius
        ddh = (2.0 * x**2 + s**2) / radius
        dddh = x * (2.0 * x**2 + 3.0 * s**2) / radius**3
        return h, dh, ddh, dddh

    def _pair(self, separation):
        sigma = self.pair_width
        gaussian = np.exp(-0.5 * (separation / sigma) ** 2)
        return gaussian

    def jastrow_basis_derivatives(self, x1, x2):
        r"""Return $B_a$, $\nabla B_a$, and $\nabla^2 B_a$.

        The amplitude is linear in positive coefficients,
        $A=aB_0+bB_1+cB_2$.
        """
        x1, x2 = np.asarray(x1), np.asarray(x2)
        h1, dh1, ddh1, _ = self._envelope(x1)
        h2, dh2, ddh2, _ = self._envelope(x2)
        separation = x1 - x2
        gaussian = self._pair(separation)
        sigma = self.pair_width
        values = np.stack(
            (
                -0.5 * (x1**2 + x2**2),
                -(h1 + h2),
                -gaussian,
            ),
            axis=-1,
        )
        gradients = np.stack(
            (
                np.stack((-x1, -x2), axis=-1),
                np.stack((-dh1, -dh2), axis=-1),
                np.stack(
                    (
                        separation * gaussian / sigma**2,
                        -separation * gaussian / sigma**2,
                    ),
                    axis=-1,
                ),
            ),
            axis=-1,
        )
        laplacians = np.stack(
            (
                -2.0 * np.ones_like(x1),
                -(ddh1 + ddh2),
                2.0
                * gaussian
                * (1.0 / sigma**2 - separation**2 / sigma**4),
            ),
            axis=-1,
        )
        return values, gradients, laplacians

    def reconstruct_parameters(self, trajectories, *, regularization=None):
        r"""Reconstruct $(a,b,c)$ from fixed-weight Stein identities.

        For $\rho=\psi^2$ and vanishing boundary terms,

        $$
        \langle \mathbf f\cdot\nabla\log\psi\rangle_{\rho}
        =-\frac12\langle\nabla\cdot\mathbf f\rangle_{\rho}.
        $$

        Taking $\mathbf f_a=\nabla B_a$ gives a three-by-three linear system.
        """
        trajectories = np.asarray(trajectories, dtype=float)
        _, basis_gradients, basis_laplacians = self.jastrow_basis_derivatives(
            trajectories[:, 0], trajectories[:, 1]
        )
        moment = np.einsum(
            "n,nka,nkb->ab", self.weights, basis_gradients, basis_gradients
        )
        target = -0.5 * np.einsum("n,na->a", self.weights, basis_laplacians)
        if regularization is None:
            regularization = self.regularization
        scale = max(float(np.trace(moment)) / 3.0, 1.0)
        system = moment + float(regularization) * scale * np.eye(3)
        coefficients = np.linalg.solve(system, target)
        coefficients = np.maximum(coefficients, 1.0e-10)
        residual = moment @ coefficients - target
        diagnostics = {
            "moment": moment,
            "target": target,
            "residual": residual,
            "condition": float(np.linalg.cond(system)),
        }
        return np.log(coefficients), diagnostics

    def stein_reconstruction_jacobian(self, trajectories, theta=None):
        r"""Differentiate the empirical Stein equations with respect to paths.

        Returns $J_{a,n\mu}=\partial\theta_a/\partial R_{n\mu}$ together
        with reconstruction diagnostics. No spatial finite differences are
        used.
        """
        trajectories = np.asarray(trajectories, dtype=float)
        x1, x2 = trajectories[:, 0], trajectories[:, 1]
        if theta is None:
            theta, diagnostics = self.reconstruct_parameters(trajectories)
        else:
            theta = np.asarray(theta, dtype=float)
            _, diagnostics = self.reconstruct_parameters(trajectories)
        coefficients = np.exp(theta)
        _, gradients, laplacians = self.jastrow_basis_derivatives(x1, x2)

        _, _, ddh1, dddh1 = self._envelope(x1)
        _, _, ddh2, dddh2 = self._envelope(x2)
        separation = x1 - x2
        gaussian = self._pair(separation)
        sigma = self.pair_width
        pair_hessian = gaussian * (
            1.0 / sigma**2 - separation**2 / sigma**4
        )
        pair_laplacian_gradient = gaussian * (
            separation**3 / sigma**6 - 3.0 * separation / sigma**4
        )

        hessians = np.zeros((self.ntraj, 3, 2, 2))
        hessians[:, 0, 0, 0] = -1.0
        hessians[:, 0, 1, 1] = -1.0
        hessians[:, 1, 0, 0] = -ddh1
        hessians[:, 1, 1, 1] = -ddh2
        hessians[:, 2, 0, 0] = pair_hessian
        hessians[:, 2, 0, 1] = -pair_hessian
        hessians[:, 2, 1, 0] = -pair_hessian
        hessians[:, 2, 1, 1] = pair_hessian

        laplacian_gradients = np.empty((self.ntraj, 3, 2))
        laplacian_gradients[:, 0, :] = 0.0
        laplacian_gradients[:, 1, 0] = -dddh1
        laplacian_gradients[:, 1, 1] = -dddh2
        laplacian_gradients[:, 2, 0] = 2.0 * pair_laplacian_gradient
        laplacian_gradients[:, 2, 1] = -2.0 * pair_laplacian_gradient

        derivative_moment = np.einsum(
            "nakm,nkb->nmab", hessians, gradients
        ) + np.einsum("nka,nbkm->nmab", gradients, hessians)
        derivative_target = -0.5 * np.swapaxes(laplacian_gradients, 1, 2)
        implicit_rhs = derivative_target - np.einsum(
            "nmab,b->nma", derivative_moment, coefficients
        )
        implicit_rhs *= self.weights[:, None, None]

        moment = diagnostics["moment"]
        scale = max(float(np.trace(moment)) / 3.0, 1.0)
        system = moment + self.regularization * scale * np.eye(3)
        coefficient_jacobian = np.linalg.solve(
            system, implicit_rhs.reshape(-1, 3).T
        ).T.reshape(self.ntraj, 2, 3)
        theta_jacobian = coefficient_jacobian / coefficients[None, None, :]
        diagnostics = dict(diagnostics)
        diagnostics["theta_jacobian"] = theta_jacobian
        return theta_jacobian, diagnostics

    def stein_tangent_lift(self, trajectories, theta=None):
        r"""Return the minimum-kinetic-energy lift $U$ with $JU=I$."""
        trajectories = np.asarray(trajectories, dtype=float)
        if theta is None:
            theta, reconstruction = self.reconstruct_parameters(trajectories)
        else:
            theta = np.asarray(theta, dtype=float)
            _, reconstruction = self.reconstruct_parameters(trajectories)
        jacobian, jacobian_diagnostics = self.stein_reconstruction_jacobian(
            trajectories, theta
        )
        inverse_mass_weights = 1.0 / (self.mass * self.weights)
        gram = np.einsum(
            "nka,n,nkb->ab", jacobian, inverse_mass_weights, jacobian
        )
        scale = max(float(np.trace(gram)) / 3.0, 1.0)
        regularized_gram = gram + self.regularization * scale * np.eye(3)
        metric = np.linalg.inv(regularized_gram)
        tangents = (
            inverse_mass_weights[:, None, None]
            * np.einsum("nkb,ba->nka", jacobian, metric)
        )
        diagnostics = dict(reconstruction)
        diagnostics.update(
            jacobian=jacobian,
            lift_identity=np.einsum("nka,nkb->ab", jacobian, tangents),
            gram=gram,
            jacobian_condition=float(np.linalg.cond(regularized_gram)),
            jacobian_diagnostics=jacobian_diagnostics,
        )
        return theta, tangents, metric, diagnostics

    def continuity_test_functions(self, trajectories, theta):
        r"""Return symmetric weak-test values and gradients.

        Polynomial invariants in center and relative coordinates probe the
        continuity equation.  The local energy is included as the final test;
        its moment makes the projected quantum force equal the negative VMC
        energy gradient on the same particle cloud.
        """
        trajectories = np.asarray(trajectories, dtype=float)
        if self.transport_basis == "local":
            values, gradients, labels = (
                self.local_transport_basis.values_and_gradients(
                    trajectories[:, :, None]
                )
            )
            return values, gradients[:, :, 0, :], labels
        if self.transport_basis == "neural":
            values, gradients = (
                self.neural_transport_model.feature_values_and_gradients(
                    trajectories[:, :, None]
                )
            )
            labels = [f"neural feature {index}" for index in range(values.shape[1])]
            return values, gradients[:, :, 0, :], labels
        x1, x2 = trajectories[:, 0], trajectories[:, 1]
        root_two = np.sqrt(2.0)
        center = (x1 + x2) / root_two
        relative = (x1 - x2) / root_two
        values, gradients, labels = [], [], []
        for degree in range(1, self.continuity_order + 1):
            for center_power in range(degree + 1):
                relative_power = degree - center_power
                center_factor = center ** (2 * center_power)
                relative_factor = relative ** (2 * relative_power)
                value = center_factor * relative_factor
                if center_power:
                    derivative_center = (
                        2
                        * center_power
                        * center ** (2 * center_power - 1)
                        * relative_factor
                    )
                else:
                    derivative_center = np.zeros_like(center)
                if relative_power:
                    derivative_relative = (
                        2
                        * relative_power
                        * center_factor
                        * relative ** (2 * relative_power - 1)
                    )
                else:
                    derivative_relative = np.zeros_like(relative)
                values.append(value)
                gradients.append(
                    np.column_stack(
                        (
                            (derivative_center + derivative_relative) / root_two,
                            (derivative_center - derivative_relative) / root_two,
                        )
                    )
                )
                labels.append(
                    f"z^{2 * center_power} r^{2 * relative_power}"
                )

        return (
            np.stack(values, axis=-1),
            np.stack(gradients, axis=-1),
            labels,
        )

    def train_neural_transport(self, trajectories, theta=None, **fit_options):
        r"""Train invariant hidden transport features on a carried cloud.

        Training shapes the nonlinear feature map with the weak-Poisson
        objective. During propagation the final readout is recomputed by the
        exact KKT constraints, so training cannot violate $JU=I$ or
        $\mathcal F=-\nabla_\theta E$.
        """
        trajectories = np.asarray(trajectories, dtype=float)
        if theta is None:
            theta, _ = self.reconstruct_parameters(trajectories)
        terms = self.log_amplitude_terms(
            trajectories[:, 0], trajectories[:, 1], theta
        )
        observables = terms["observables"].T
        scores = 2.0 * (
            observables - np.einsum("n,na->a", self.weights, observables)
        )
        self.neural_transport_model.fit(
            trajectories[:, :, None],
            scores,
            weights=self.weights,
            mass=self.mass,
            **fit_options,
        )
        return self.neural_transport_model

    def constrained_continuity_lift(self, trajectories, theta=None):
        r"""Solve a constrained Monte Carlo weak-continuity problem.

        With $phi_a=\sum_k c_{ka}f_k$ and
        $U_a=M^{-1}\nabla\phi_a$, the coefficients minimize

        $$
        \frac12 c_a^T Kc_a-b_a^Tc_a,
        \quad K_{kl}=\langle\nabla f_k^TM^{-1}\nabla f_l\rangle,
        \quad b_{ka}=\langle f_k s_a\rangle,
        $$

        subject to $JU=I$ and
        $\langle U_a\cdot(F_{\rm cl}+F_Q)\rangle=-\partial_aE$.
        Thus density transport is fitted variationally while reconstruction
        and energy descent are exact finite-cloud constraints.  The KKT system
        size depends on basis size, not particle count or spatial dimension.
        """
        trajectories = np.asarray(trajectories, dtype=float)
        if theta is None:
            theta, reconstruction = self.reconstruct_parameters(trajectories)
        else:
            theta = np.asarray(theta, dtype=float)
            _, reconstruction = self.reconstruct_parameters(trajectories)
        jacobian, jacobian_diagnostics = self.stein_reconstruction_jacobian(
            trajectories, theta
        )
        values, gradients, labels = self.continuity_test_functions(
            trajectories, theta
        )
        terms = self.log_amplitude_terms(
            trajectories[:, 0], trajectories[:, 1], theta
        )
        observables = terms["observables"].T
        scores = 2.0 * (
            observables - np.einsum("n,na->a", self.weights, observables)
        )
        centered_values = values - np.einsum(
            "n,nk->k", self.weights, values
        )[None, :]
        weak_targets = np.einsum(
            "n,nk,na->ka", self.weights, centered_values, scores
        )
        kinetic = np.einsum(
            "n,nmk,nml->kl",
            self.weights,
            gradients / self.mass,
            gradients,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(kinetic)
        largest_eigenvalue = max(float(eigenvalues[-1]), 1.0e-30)
        retained = eigenvalues > self.regularization * largest_eigenvalue
        minimum_rank = min(kinetic.shape[0], 4)
        if np.count_nonzero(retained) < minimum_rank:
            retained[-minimum_rank:] = True
        whitening = eigenvectors[:, retained] / np.sqrt(
            np.maximum(eigenvalues[retained], self.regularization * largest_eigenvalue)
        )[None, :]
        scaled_gradients = np.einsum("nmk,kl->nml", gradients, whitening)
        scaled_targets = whitening.T @ weak_targets
        scaled_kinetic = np.einsum(
            "n,nmk,nml->kl", self.weights, scaled_gradients / self.mass, scaled_gradients
        )
        reconstruction_constraints = np.einsum(
            "nma,nmk->ak", jacobian, scaled_gradients / self.mass
        )
        _, quantum_force = self.quantum_potential_force(
            trajectories[:, 0], trajectories[:, 1], theta
        )
        residual_force = self.classical_force(
            trajectories[:, 0], trajectories[:, 1]
        ) + quantum_force
        force_constraint = np.einsum(
            "n,nmk,nm->k",
            self.weights,
            scaled_gradients / self.mass,
            residual_force,
        )
        constraints = np.vstack(
            (reconstruction_constraints, force_constraint[None, :])
        )
        energy_gradient = self._sample_state(
            theta, trajectories, with_tangents=False
        )["gradient"]
        constraint_targets = np.vstack((np.eye(3), -energy_gradient[None, :]))

        basis_size = scaled_kinetic.shape[0]
        constraint_size = constraints.shape[0]
        kkt = np.block(
            [
                [
                    scaled_kinetic
                    + self.regularization * np.eye(basis_size),
                    constraints.T,
                ],
                [constraints, np.zeros((constraint_size, constraint_size))],
            ]
        )
        right_hand_side = np.vstack(
            (scaled_targets, constraint_targets)
        )
        try:
            solution = np.linalg.solve(kkt, right_hand_side)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(
                kkt, right_hand_side, rcond=self.regularization
            )[0]
        whitened_coefficients = solution[:basis_size]
        coefficients = whitening @ whitened_coefficients
        multipliers = solution[basis_size:]
        tangents = np.einsum(
            "nmk,ka->nma", gradients / self.mass, coefficients
        )
        lift_identity = np.einsum("nma,nmb->ab", jacobian, tangents)
        force_prediction = np.einsum(
            "n,nma,nm->a", self.weights, tangents, residual_force
        )
        weak_residual = kinetic @ coefficients - weak_targets
        optimality_residual = (
            whitening.T @ weak_residual + constraints.T @ multipliers
        )
        metric = self.mass * np.einsum(
            "n,nka,nkb->ab", self.weights, tangents, tangents
        )
        diagnostics = dict(reconstruction)
        diagnostics.update(
            jacobian=jacobian,
            lift_identity=lift_identity,
            weak_residual=weak_residual,
            weak_targets=weak_targets,
            test_labels=labels,
            force_prediction=force_prediction,
            optimality_residual=optimality_residual,
            constraint_condition=float(np.linalg.cond(kkt)),
            retained_basis_rank=int(np.count_nonzero(retained)),
            kinetic_eigenvalues=eigenvalues,
            jacobian_diagnostics=jacobian_diagnostics,
        )
        return theta, tangents, metric, diagnostics

    def log_amplitude_terms(self, x1, x2, theta=None):
        """Return amplitude derivatives needed for energy and quantum force."""
        if theta is None:
            theta = self.theta
        a, b, c = np.exp(np.asarray(theta, dtype=float))
        x1, x2 = np.asarray(x1), np.asarray(x2)
        h1, dh1, ddh1, dddh1 = self._envelope(x1)
        h2, dh2, ddh2, dddh2 = self._envelope(x2)
        separation = x1 - x2
        gaussian = self._pair(separation)
        sigma = self.pair_width
        pair = -c * gaussian
        dpair = c * separation * gaussian / sigma**2
        ddpair = c * gaussian * (1.0 / sigma**2 - separation**2 / sigma**4)
        dddpair = c * gaussian * (
            separation**3 / sigma**6 - 3.0 * separation / sigma**4
        )
        f1 = -0.5 * a * x1**2 - b * h1
        f2 = -0.5 * a * x2**2 - b * h2
        df1, df2 = -a * x1 - b * dh1, -a * x2 - b * dh2
        ddf1, ddf2 = -a - b * ddh1, -a - b * ddh2
        dddf1, dddf2 = -b * dddh1, -b * dddh2
        return {
            "amplitude": f1 + f2 + pair,
            "d1": df1 + dpair,
            "d2": df2 - dpair,
            "dd1": ddf1 + ddpair,
            "dd2": ddf2 + ddpair,
            "ddd1": dddf1 + 2.0 * dddpair,
            "ddd2": dddf2 - 2.0 * dddpair,
            "mixed": -ddpair,
            "observables": np.stack(
                (-0.5 * a * (x1**2 + x2**2), -b * (h1 + h2), pair), axis=0
            ),
        }

    def potential(self, x1, x2):
        return (
            quartic_potential(x1, self.anharmonicity)
            + quartic_potential(x2, self.anharmonicity)
            + soft_coulomb(x1, x2, self.interaction, self.softening)
        )

    def classical_force(self, x1, x2):
        separation = np.asarray(x1) - np.asarray(x2)
        interaction_force = self.interaction * separation / (
            separation**2 + self.softening**2
        ) ** 1.5
        force1 = -np.asarray(x1) - self.anharmonicity * np.asarray(x1) ** 3
        force2 = -np.asarray(x2) - self.anharmonicity * np.asarray(x2) ** 3
        return np.stack((force1 + interaction_force, force2 - interaction_force), axis=-1)

    def _quantum_potential_force_analytic(self, x1, x2, theta):
        terms = self.log_amplitude_terms(x1, x2, theta)
        prefactor = self.hbar**2 / (2.0 * self.mass)
        quantum_potential = -prefactor * (
            terms["dd1"] + terms["dd2"] + terms["d1"] ** 2 + terms["d2"] ** 2
        )
        force1 = prefactor * (
            terms["ddd1"]
            + 2.0
            * (
                terms["d1"] * terms["dd1"]
                + terms["d2"] * terms["mixed"]
            )
        )
        force2 = prefactor * (
            terms["ddd2"]
            + 2.0
            * (
                terms["d1"] * terms["mixed"]
                + terms["d2"] * terms["dd2"]
            )
        )
        return quantum_potential, np.stack((force1, force2), axis=-1)

    def _build_ad_point_kernel(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError(
                "The AD quantum-force backend requires JAX; install pyqed[ml]."
            ) from error

        jax.config.update("jax_enable_x64", True)
        smoothing = self.envelope_smoothing
        pair_width = self.pair_width
        mass = self.mass
        hbar = self.hbar

        def log_amplitude(theta, coordinates):
            a, b, c = jnp.exp(theta)
            x1, x2 = coordinates
            radius1 = jnp.sqrt(x1**2 + smoothing**2)
            radius2 = jnp.sqrt(x2**2 + smoothing**2)
            h1 = (radius1**3 - smoothing**3) / 3.0
            h2 = (radius2**3 - smoothing**3) / 3.0
            separation = x1 - x2
            pair = -c * jnp.exp(-0.5 * (separation / pair_width) ** 2)
            return -0.5 * a * (x1**2 + x2**2) - b * (h1 + h2) + pair

        def quantum_potential(theta, coordinates):
            gradient = jax.grad(log_amplitude, argnums=1)(theta, coordinates)
            hessian = jax.hessian(log_amplitude, argnums=1)(theta, coordinates)
            return -(hbar**2 / (2.0 * mass)) * (
                jnp.trace(hessian) + jnp.dot(gradient, gradient)
            )

        def point_kernel(theta, coordinates):
            potential, potential_gradient = jax.value_and_grad(
                quantum_potential, argnums=1
            )(theta, coordinates)
            return potential, -potential_gradient

        self._ad_point_kernel = jax.jit(jax.vmap(point_kernel, in_axes=(None, 0)))

    def _quantum_potential_force_ad(self, x1, x2, theta):
        if self._ad_point_kernel is None:
            self._build_ad_point_kernel()
        coordinates = np.column_stack(
            (np.asarray(x1, dtype=float).ravel(), np.asarray(x2, dtype=float).ravel())
        )
        potential, force = self._ad_point_kernel(np.asarray(theta), coordinates)
        shape = np.broadcast_shapes(np.shape(x1), np.shape(x2))
        return np.asarray(potential).reshape(shape), np.asarray(force).reshape(shape + (2,))

    def quantum_potential_force(self, x1, x2, theta=None, *, backend=None):
        r"""Evaluate $Q$ and $-\nabla Q$ analytically or with JAX AD."""
        if theta is None:
            theta = self.theta
        if backend is None:
            backend = self.force_backend
        if backend == "analytic":
            return self._quantum_potential_force_analytic(x1, x2, theta)
        if backend == "ad":
            return self._quantum_potential_force_ad(x1, x2, theta)
        raise ValueError("backend must be 'analytic' or 'ad'")

    def _density_grid(self, theta):
        x1, x2 = np.meshgrid(self.grid, self.grid, indexing="ij")
        terms = self.log_amplitude_terms(x1, x2, theta)
        amplitude = terms["amplitude"]
        shifted = amplitude - np.max(amplitude)
        rho = np.exp(2.0 * shifted)
        rho /= np.trapezoid(np.trapezoid(rho, self.grid, axis=1), self.grid)
        return rho

    def _grid_state(self, theta):
        x1, x2 = np.meshgrid(self.grid, self.grid, indexing="ij")
        rho = self._density_grid(theta)
        terms = self.log_amplitude_terms(x1, x2, theta)
        quantum_potential, _ = self.quantum_potential_force(
            x1, x2, theta, backend="analytic"
        )
        local_energy = self.potential(x1, x2) + quantum_potential
        energy = float(
            np.trapezoid(np.trapezoid(rho * local_energy, self.grid, axis=1), self.grid)
        )
        variance = float(
            np.trapezoid(
                np.trapezoid(rho * (local_energy - energy) ** 2, self.grid, axis=1),
                self.grid,
            )
        )
        means = np.array(
            [
                np.trapezoid(
                    np.trapezoid(rho * observable, self.grid, axis=1), self.grid
                )
                for observable in terms["observables"]
            ]
        )
        gradient = np.array(
            [
                2.0
                * np.trapezoid(
                    np.trapezoid(
                        rho * (observable - mean) * (local_energy - energy),
                        self.grid,
                        axis=1,
                    ),
                    self.grid,
                )
                for observable, mean in zip(terms["observables"], means)
            ]
        )
        return rho, energy, variance, gradient

    def _transport(self, rho):
        marginal = np.trapezoid(rho, self.grid, axis=1)
        cdf1 = cumulative_trapezoid(marginal, self.grid, initial=0.0)
        cdf1 /= cdf1[-1]
        trajectory_x1 = np.interp(self.quantiles[:, 0], cdf1, self.grid)
        indices = np.searchsorted(self.grid, trajectory_x1, side="right") - 1
        indices = np.clip(indices, 0, self.grid.size - 2)
        fractions = (trajectory_x1 - self.grid[indices]) / (
            self.grid[indices + 1] - self.grid[indices]
        )
        conditional = (
            (1.0 - fractions[:, None]) * rho[indices]
            + fractions[:, None] * rho[indices + 1]
        )
        cdf2 = cumulative_trapezoid(conditional, self.grid, axis=1, initial=0.0)
        cdf2 /= cdf2[:, -1, None]
        targets = self.quantiles[:, 1]
        lower = np.sum(cdf2 <= targets[:, None], axis=1) - 1
        lower = np.clip(lower, 0, self.grid.size - 2)
        rows = np.arange(self.ntraj)
        cdf_lower = cdf2[rows, lower]
        cdf_upper = cdf2[rows, lower + 1]
        fractions2 = (targets - cdf_lower) / np.maximum(cdf_upper - cdf_lower, 1.0e-15)
        trajectory_x2 = self.grid[lower] + fractions2 * (
            self.grid[lower + 1] - self.grid[lower]
        )
        return np.column_stack((trajectory_x1, trajectory_x2))

    def sample_initial(self, theta):
        """Sample the initial Jastrow density using the fixed random labels."""
        return self._transport(self._density_grid(np.asarray(theta, dtype=float)))

    def _transport_tangent_fields(self, theta):
        tangents = np.empty((self.ntraj, 2, 3))
        for parameter in range(3):
            displacement = np.zeros(3)
            displacement[parameter] = self.tangent_step
            plus = self._transport(self._density_grid(theta + displacement))
            minus = self._transport(self._density_grid(theta - displacement))
            tangents[:, :, parameter] = (plus - minus) / (2.0 * self.tangent_step)
        return tangents

    def _sample_state(
        self, theta, trajectories, with_tangents=True, tangent_data=None
    ):
        trajectories = np.asarray(trajectories, dtype=float)
        terms = self.log_amplitude_terms(
            trajectories[:, 0], trajectories[:, 1], theta
        )
        quantum_potential, quantum_force = self.quantum_potential_force(
            trajectories[:, 0], trajectories[:, 1], theta
        )
        local_energy = self.potential(
            trajectories[:, 0], trajectories[:, 1]
        ) + quantum_potential
        energy = float(np.dot(self.weights, local_energy))
        variance = float(np.dot(self.weights, (local_energy - energy) ** 2))
        observable_means = terms["observables"] @ self.weights
        gradient = 2.0 * np.sum(
            (terms["observables"] - observable_means[:, None])
            * (local_energy - energy)[None, :]
            * self.weights[None, :],
            axis=1,
        )
        state = {
            "energy": energy,
            "variance": variance,
            "gradient": gradient,
            "local_energy": local_energy,
            "x": trajectories,
        }
        if not with_tangents:
            return state
        if self.tangent_backend in {"continuity", "stein"}:
            if tangent_data is None:
                if self.tangent_backend == "continuity":
                    _, tangents, metric, tangent_diagnostics = (
                        self.constrained_continuity_lift(trajectories, theta)
                    )
                else:
                    _, tangents, metric, tangent_diagnostics = (
                        self.stein_tangent_lift(trajectories, theta)
                    )
            else:
                tangents, metric, tangent_diagnostics = tangent_data
        else:
            tangents = self._transport_tangent_fields(theta)
            metric = self.mass * np.einsum(
                "n,nka,nkb->ab", self.weights, tangents, tangents
            )
            tangent_diagnostics = None
        residual_force = self.classical_force(
            trajectories[:, 0], trajectories[:, 1]
        ) + quantum_force
        sampled_force = np.einsum(
            "n,nka,nk->a", self.weights, tangents, residual_force
        )
        state.update(
            tangents=tangents,
            metric=metric,
            force=sampled_force,
            sampled_force=sampled_force,
            residual_force=residual_force,
            tangent_diagnostics=tangent_diagnostics,
        )
        return state

    def _state(self, theta, with_tangents=True):
        """Evaluate a freshly transported equilibrium sample for diagnostics."""
        trajectories = self._transport(self._density_grid(theta))
        state = self._sample_state(theta, trajectories, with_tangents)
        state["rho"] = self._density_grid(theta)
        return state

    def energy_at(self, theta):
        return self._state(np.asarray(theta, dtype=float), with_tangents=False)[
            "energy"
        ]

    def run(
        self,
        *,
        theta0=(np.log(1.2), np.log(0.5), np.log(0.5)),
        dt=0.04,
        max_steps=300,
        tolerance=2.0e-5,
        record_every=2,
        max_parameter_step=0.025,
        parameter_closure="stein",
    ):
        """Relax explicitly propagated, fixed-weight trajectories.

        With ``parameter_closure='stein'``, coordinates are the only integrated
        state and Jastrow parameters are reconstructed at every Runge--Kutta
        stage. ``'coupled'`` retains explicit parameter integration as a
        reference implementation.
        """
        if parameter_closure == "stein":
            return self._run_stein(
                theta0=theta0,
                dt=dt,
                max_steps=max_steps,
                tolerance=tolerance,
                record_every=record_every,
                max_parameter_step=max_parameter_step,
            )
        if parameter_closure != "coupled":
            raise ValueError("parameter_closure must be 'stein' or 'coupled'")
        theta = np.asarray(theta0, dtype=float).copy()
        trajectories_current = self.sample_initial(theta)
        self.initial_x = trajectories_current.copy()
        time = 0.0
        times, energies, variances, parameters, gradients, forces, trajectories, drifts = (
            [], [], [], [], [], [], [], []
        )
        for step_index in range(max_steps + 1):
            state = self._sample_state(theta, trajectories_current)
            velocity, trajectory_velocity = self._flow_rhs(
                theta, trajectories_current, state
            )
            if step_index % record_every == 0 or step_index == max_steps:
                transported = self._transport(self._density_grid(theta))
                drift = np.sqrt(
                    np.sum(
                        self.weights[:, None]
                        * (trajectories_current - transported) ** 2
                    )
                )
                times.append(time)
                energies.append(state["energy"])
                variances.append(state["variance"])
                parameters.append(theta.copy())
                gradients.append(state["gradient"].copy())
                forces.append(state["force"].copy())
                trajectories.append(trajectories_current.copy())
                drifts.append(drift)
            if np.linalg.norm(state["force"], ord=np.inf) < tolerance:
                self.success = True
                self.message = "sampled projected force converged"
                break
            if step_index == max_steps:
                self.message = "maximum steps reached"
                break
            trial_dt = min(
                float(dt),
                max_parameter_step / max(np.linalg.norm(velocity, ord=np.inf), 1.0e-15),
            )
            theta, trajectories_current = self._rk4_step(
                theta, trajectories_current, trial_dt, state
            )
            if not (
                np.all(np.isfinite(theta))
                and np.all(np.isfinite(trajectories_current))
                and np.all(np.abs(theta) < 20.0)
            ):
                self.message = "trajectory integration became non-finite"
                break
            time += trial_dt
        final_state = self._sample_state(theta, trajectories_current)
        _, reference_energy, _, reference_gradient = self._grid_state(theta)
        transported = self._transport(self._density_grid(theta))
        self.theta = theta
        self.x = trajectories_current
        self.energy = final_state["energy"]
        self.gradient = final_state["gradient"]
        self.reference_energy = reference_energy
        self.reference_gradient = reference_gradient
        self.transport_drift = float(
            np.sqrt(
                np.sum(
                    self.weights[:, None] * (trajectories_current - transported) ** 2
                )
            )
        )
        self.history = {
            "time": np.asarray(times),
            "energy": np.asarray(energies),
            "variance": np.asarray(variances),
            "theta": np.asarray(parameters),
            "gradient": np.asarray(gradients),
            "projected_force": np.asarray(forces),
            "x": np.asarray(trajectories),
            "transport_drift": np.asarray(drifts),
        }
        return self

    def _stein_flow_rhs(self, trajectories):
        if self.tangent_backend == "continuity":
            theta, tangents, metric, lift_diagnostics = (
                self.constrained_continuity_lift(trajectories)
            )
        else:
            theta, tangents, metric, lift_diagnostics = self.stein_tangent_lift(
                trajectories
            )
        state = self._sample_state(
            theta,
            trajectories,
            tangent_data=(tangents, metric, lift_diagnostics),
        )
        parameter_velocity, trajectory_velocity = self._flow_rhs(
            theta, trajectories, state
        )
        kinematic_velocity = np.einsum(
            "nka,nk->a", lift_diagnostics["jacobian"], trajectory_velocity
        )
        lift_diagnostics["kinematic_velocity"] = kinematic_velocity
        lift_diagnostics["parameter_velocity"] = parameter_velocity
        return (
            trajectory_velocity,
            theta,
            parameter_velocity,
            state,
            lift_diagnostics,
        )

    def _stein_rk4_step(self, trajectories, dt, first_rhs):
        k1 = first_rhs
        k2 = self._stein_flow_rhs(trajectories + 0.5 * dt * k1)[0]
        k3 = self._stein_flow_rhs(trajectories + 0.5 * dt * k2)[0]
        k4 = self._stein_flow_rhs(trajectories + dt * k3)[0]
        return trajectories + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _run_stein(
        self,
        *,
        theta0,
        dt,
        max_steps,
        tolerance,
        record_every,
        max_parameter_step,
    ):
        initial_theta = np.asarray(theta0, dtype=float)
        trajectories_current = self.sample_initial(initial_theta)
        self.initial_x = trajectories_current.copy()
        time = 0.0
        times, energies, variances, parameters, gradients, forces, trajectories = (
            [], [], [], [], [], [], []
        )
        drifts, stein_residuals, stein_conditions = [], [], []
        parameter_velocities, kinematic_errors, lift_errors = [], [], []
        continuity_residuals, force_gradient_gaps = [], []

        for step_index in range(max_steps + 1):
            (
                trajectory_velocity,
                theta,
                parameter_velocity,
                state,
                diagnostics,
            ) = self._stein_flow_rhs(trajectories_current)
            converged = (
                np.linalg.norm(parameter_velocity, ord=np.inf) < tolerance
                and np.linalg.norm(state["gradient"], ord=np.inf) < tolerance
                and np.linalg.norm(
                    state["force"] + state["gradient"], ord=np.inf
                )
                < tolerance
            )
            if step_index % record_every == 0 or step_index == max_steps or converged:
                transported = self._transport(self._density_grid(theta))
                drift = np.sqrt(
                    np.sum(
                        self.weights[:, None]
                        * (trajectories_current - transported) ** 2
                    )
                )
                times.append(time)
                energies.append(state["energy"])
                variances.append(state["variance"])
                parameters.append(theta.copy())
                gradients.append(state["gradient"].copy())
                forces.append(state["force"].copy())
                trajectories.append(trajectories_current.copy())
                drifts.append(drift)
                stein_residuals.append(diagnostics["residual"].copy())
                stein_conditions.append(diagnostics["condition"])
                parameter_velocities.append(parameter_velocity.copy())
                kinematic_errors.append(
                    (diagnostics["kinematic_velocity"] - parameter_velocity).copy()
                )
                lift_errors.append(
                    float(
                        np.max(
                            np.abs(diagnostics["lift_identity"] - np.eye(3))
                        )
                    )
                )
                if "weak_residual" in diagnostics:
                    continuity_residuals.append(
                        diagnostics["weak_residual"].copy()
                    )
                else:
                    continuity_residuals.append(np.full((0, 3), np.nan))
                force_gradient_gaps.append(
                    (state["force"] + state["gradient"]).copy()
                )
            if converged:
                self.success = True
                if self.tangent_backend == "continuity":
                    self.message = "constrained continuity flow converged"
                else:
                    self.message = "Stein-closed trajectory velocity converged"
                break
            if step_index == max_steps:
                self.message = "maximum steps reached"
                break
            step = min(
                float(dt),
                max_parameter_step
                / max(np.linalg.norm(parameter_velocity, ord=np.inf), 1.0e-15),
            )
            trajectories_current = self._stein_rk4_step(
                trajectories_current, step, trajectory_velocity
            )
            if not np.all(np.isfinite(trajectories_current)):
                self.message = "trajectory integration became non-finite"
                break
            time += step

        (
            _,
            theta,
            parameter_velocity,
            final_state,
            diagnostics,
        ) = self._stein_flow_rhs(trajectories_current)
        _, reference_energy, _, reference_gradient = self._grid_state(theta)
        transported = self._transport(self._density_grid(theta))
        self.theta = theta
        self.x = trajectories_current
        self.energy = final_state["energy"]
        self.gradient = final_state["gradient"]
        self.reference_energy = reference_energy
        self.reference_gradient = reference_gradient
        self.transport_drift = float(
            np.sqrt(
                np.sum(
                    self.weights[:, None]
                    * (trajectories_current - transported) ** 2
                )
            )
        )
        self.stein_residual = diagnostics["residual"]
        self.stein_condition = diagnostics["condition"]
        self.parameter_velocity = parameter_velocity
        self.kinematic_error = diagnostics["kinematic_velocity"] - parameter_velocity
        self.lift_error = float(
            np.max(np.abs(diagnostics["lift_identity"] - np.eye(3)))
        )
        self.continuity_residual = diagnostics.get("weak_residual")
        self.force_gradient_gap = final_state["force"] + final_state["gradient"]
        self.history = {
            "time": np.asarray(times),
            "energy": np.asarray(energies),
            "variance": np.asarray(variances),
            "theta": np.asarray(parameters),
            "gradient": np.asarray(gradients),
            "projected_force": np.asarray(forces),
            "x": np.asarray(trajectories),
            "transport_drift": np.asarray(drifts),
            "stein_residual": np.asarray(stein_residuals),
            "stein_condition": np.asarray(stein_conditions),
            "parameter_velocity": np.asarray(parameter_velocities),
            "kinematic_error": np.asarray(kinematic_errors),
            "lift_error": np.asarray(lift_errors),
            "continuity_residual": np.asarray(continuity_residuals),
            "force_gradient_gap": np.asarray(force_gradient_gaps),
        }
        return self

    def _flow_rhs(self, theta, trajectories, state=None):
        if state is None:
            state = self._sample_state(theta, trajectories)
        scale = max(float(np.trace(state["metric"])) / 3.0, 1.0)
        metric = state["metric"] + self.regularization * scale * np.eye(3)
        parameter_velocity = np.linalg.solve(
            self.friction * metric, state["force"]
        )
        trajectory_velocity = np.einsum(
            "nka,a->nk", state["tangents"], parameter_velocity
        )
        return parameter_velocity, trajectory_velocity

    def _rk4_step(self, theta, trajectories, dt, initial_state):
        k1_theta, k1_x = self._flow_rhs(theta, trajectories, initial_state)
        k2_theta, k2_x = self._flow_rhs(
            theta + 0.5 * dt * k1_theta,
            trajectories + 0.5 * dt * k1_x,
        )
        k3_theta, k3_x = self._flow_rhs(
            theta + 0.5 * dt * k2_theta,
            trajectories + 0.5 * dt * k2_x,
        )
        k4_theta, k4_x = self._flow_rhs(
            theta + dt * k3_theta,
            trajectories + dt * k3_x,
        )
        return (
            theta + dt * (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta) / 6.0,
            trajectories + dt * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0,
        )

    def density(self):
        if self.theta is None:
            raise RuntimeError("run the solver before requesting its density")
        return self._density_grid(self.theta)
