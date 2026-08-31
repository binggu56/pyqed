"""Three interacting particles in one dimension with real trajectory flow."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import eigsh

from .jastrow_1d import quartic_potential
from .transport_basis import SharedRadialTransportBasis


def exact_three_particle_ground_state(
    *,
    anharmonicity=4.0,
    interaction=1.0,
    softening=0.5,
    mass=1.0,
    hbar=1.0,
    xmax=3.8,
    ngrid=35,
):
    """Return the positive bosonic ground state on a sparse 3D grid."""
    if ngrid < 13 or ngrid % 2 == 0:
        raise ValueError("ngrid must be an odd integer of at least thirteen")
    full_grid = np.linspace(-xmax, xmax, ngrid)
    dx = full_grid[1] - full_grid[0]
    grid = full_grid[1:-1]
    size = grid.size
    kinetic_1d = -(hbar**2 / (2.0 * mass * dx**2)) * diags(
        (np.ones(size - 1), -2.0 * np.ones(size), np.ones(size - 1)),
        (-1, 0, 1),
        format="csr",
    )
    identity = eye(size, format="csr")
    kinetic = (
        kron(kron(kinetic_1d, identity), identity, format="csr")
        + kron(kron(identity, kinetic_1d), identity, format="csr")
        + kron(kron(identity, identity), kinetic_1d, format="csr")
    )
    x1, x2, x3 = np.meshgrid(grid, grid, grid, indexing="ij")
    coordinates = np.stack((x1, x2, x3), axis=-1)
    potential = np.sum(quartic_potential(coordinates, anharmonicity), axis=-1)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        separation = coordinates[..., first] - coordinates[..., second]
        potential += interaction / np.sqrt(separation**2 + softening**2)
    hamiltonian = kinetic + diags(potential.ravel(), format="csr")
    energy, vector = eigsh(hamiltonian, k=1, which="SA", tol=1.0e-10)
    psi = vector[:, 0].reshape((size, size, size))
    psi /= np.sqrt(np.sum(psi**2) * dx**3)
    if psi[size // 2, size // 2, size // 2] < 0.0:
        psi *= -1.0
    return grid, psi, float(energy[0])


class ProjectedThreeParticleJastrow1D:
    r"""Constrained dissipative trajectories for three particles in 1D."""

    nparticles = 3
    nparameters = 3

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
        xmax=3.8,
        ngrid=41,
        regularization=1.0e-9,
        transport_basis=None,
    ):
        if ntraj < 64:
            raise ValueError("ntraj must be at least 64")
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
        self.regularization = float(regularization)
        self.grid = np.linspace(-xmax, xmax, int(ngrid))
        self.weights = np.full(self.ntraj, 1.0 / self.ntraj)
        self.transport_basis = transport_basis or SharedRadialTransportBasis()
        self.theta = self.x = self.energy = self.reference_energy = None
        self.gradient = self.reference_gradient = None
        self.history = None
        self.success = False
        self.message = "not run"
        self.acceptance = None

    def _envelope(self, x):
        smoothing = self.envelope_smoothing
        radius = np.sqrt(x**2 + smoothing**2)
        return (
            (radius**3 - smoothing**3) / 3.0,
            x * radius,
            (2.0 * x**2 + smoothing**2) / radius,
            x * (2.0 * x**2 + 3.0 * smoothing**2) / radius**3,
        )

    def log_amplitude_terms(self, coordinates, theta):
        coordinates = np.asarray(coordinates, dtype=float)
        coefficients = np.exp(np.asarray(theta, dtype=float))
        a, b, c = coefficients
        envelope, derivative, second, third = self._envelope(coordinates)
        amplitude = -0.5 * a * np.sum(coordinates**2, axis=-1) - b * np.sum(
            envelope, axis=-1
        )
        gradient = -a * coordinates - b * derivative
        diagonal_hessian = -a - b * second
        trace_gradient = -b * third
        hessian = np.zeros(coordinates.shape + (self.nparticles,))
        diagonal = np.arange(self.nparticles)
        hessian[..., diagonal, diagonal] = diagonal_hessian
        pair_sum = np.zeros(coordinates.shape[:-1])
        sigma = self.pair_width
        for first, second_index in ((0, 1), (0, 2), (1, 2)):
            separation = coordinates[..., first] - coordinates[..., second_index]
            gaussian = np.exp(-0.5 * (separation / sigma) ** 2)
            pair_sum += gaussian
            amplitude -= c * gaussian
            first_derivative = c * separation * gaussian / sigma**2
            pair_second = c * gaussian * (
                1.0 / sigma**2 - separation**2 / sigma**4
            )
            pair_third = c * gaussian * (
                separation**3 / sigma**6 - 3.0 * separation / sigma**4
            )
            gradient[..., first] += first_derivative
            gradient[..., second_index] -= first_derivative
            hessian[..., first, first] += pair_second
            hessian[..., second_index, second_index] += pair_second
            hessian[..., first, second_index] -= pair_second
            hessian[..., second_index, first] -= pair_second
            trace_gradient[..., first] += 2.0 * pair_third
            trace_gradient[..., second_index] -= 2.0 * pair_third
        observables = np.stack(
            (
                -0.5 * a * np.sum(coordinates**2, axis=-1),
                -b * np.sum(envelope, axis=-1),
                -c * pair_sum,
            ),
            axis=-1,
        )
        return amplitude, gradient, hessian, trace_gradient, observables

    def potential(self, coordinates):
        coordinates = np.asarray(coordinates)
        value = np.sum(
            quartic_potential(coordinates, self.anharmonicity), axis=-1
        )
        for first, second in ((0, 1), (0, 2), (1, 2)):
            separation = coordinates[..., first] - coordinates[..., second]
            value += self.interaction / np.sqrt(
                separation**2 + self.softening**2
            )
        return value

    def classical_force(self, coordinates):
        coordinates = np.asarray(coordinates)
        force = -coordinates - self.anharmonicity * coordinates**3
        for first, second in ((0, 1), (0, 2), (1, 2)):
            separation = coordinates[..., first] - coordinates[..., second]
            pair_force = self.interaction * separation / (
                separation**2 + self.softening**2
            ) ** 1.5
            force[..., first] += pair_force
            force[..., second] -= pair_force
        return force

    def quantum_potential_force(self, coordinates, theta):
        _, gradient, hessian, trace_gradient, _ = self.log_amplitude_terms(
            coordinates, theta
        )
        prefactor = self.hbar**2 / (2.0 * self.mass)
        trace = np.trace(hessian, axis1=-2, axis2=-1)
        quantum_potential = -prefactor * (
            trace + np.sum(gradient**2, axis=-1)
        )
        quantum_force = prefactor * (
            trace_gradient + 2.0 * np.einsum("...i,...ik->...k", gradient, hessian)
        )
        return quantum_potential, quantum_force

    def jastrow_basis_derivatives(self, coordinates):
        coordinates = np.asarray(coordinates)
        envelope, derivative, second, third = self._envelope(coordinates)
        values = np.stack(
            (
                -0.5 * np.sum(coordinates**2, axis=-1),
                -np.sum(envelope, axis=-1),
                np.zeros(coordinates.shape[:-1]),
            ),
            axis=-1,
        )
        gradients = np.zeros(coordinates.shape + (3,))
        gradients[..., 0] = -coordinates
        gradients[..., 1] = -derivative
        laplacians = np.stack(
            (
                -self.nparticles * np.ones(coordinates.shape[:-1]),
                -np.sum(second, axis=-1),
                np.zeros(coordinates.shape[:-1]),
            ),
            axis=-1,
        )
        hessians = np.zeros(
            coordinates.shape[:-1] + (3, self.nparticles, self.nparticles)
        )
        diagonal = np.arange(self.nparticles)
        hessians[..., 0, diagonal, diagonal] = -1.0
        hessians[..., 1, diagonal, diagonal] = -second
        laplacian_gradients = np.zeros(coordinates.shape[:-1] + (3, self.nparticles))
        laplacian_gradients[..., 1, :] = -third
        sigma = self.pair_width
        pair_value = np.zeros(coordinates.shape[:-1])
        for first, second_index in ((0, 1), (0, 2), (1, 2)):
            separation = coordinates[..., first] - coordinates[..., second_index]
            gaussian = np.exp(-0.5 * (separation / sigma) ** 2)
            pair_value += gaussian
            first_derivative = separation * gaussian / sigma**2
            pair_second = gaussian * (
                1.0 / sigma**2 - separation**2 / sigma**4
            )
            pair_third = gaussian * (
                separation**3 / sigma**6 - 3.0 * separation / sigma**4
            )
            gradients[..., first, 2] += first_derivative
            gradients[..., second_index, 2] -= first_derivative
            laplacians[..., 2] += 2.0 * pair_second
            hessians[..., 2, first, first] += pair_second
            hessians[..., 2, second_index, second_index] += pair_second
            hessians[..., 2, first, second_index] -= pair_second
            hessians[..., 2, second_index, first] -= pair_second
            laplacian_gradients[..., 2, first] += 2.0 * pair_third
            laplacian_gradients[..., 2, second_index] -= 2.0 * pair_third
        values[..., 2] = -pair_value
        return values, gradients, laplacians, hessians, laplacian_gradients

    def reconstruct_parameters(self, trajectories):
        _, gradients, laplacians, _, _ = self.jastrow_basis_derivatives(
            trajectories
        )
        moment = np.einsum(
            "n,nia,nib->ab", self.weights, gradients, gradients
        )
        target = -0.5 * np.einsum("n,na->a", self.weights, laplacians)
        scale = max(float(np.trace(moment)) / 3.0, 1.0)
        system = moment + self.regularization * scale * np.eye(3)
        coefficients = np.maximum(np.linalg.solve(system, target), 1.0e-10)
        return np.log(coefficients), {
            "moment": moment,
            "target": target,
            "residual": moment @ coefficients - target,
            "condition": float(np.linalg.cond(system)),
        }

    def reconstruction_jacobian(self, trajectories, theta):
        coefficients = np.exp(theta)
        _, gradients, _, hessians, laplacian_gradients = (
            self.jastrow_basis_derivatives(trajectories)
        )
        derivative_moment = np.einsum(
            "nakm,nkb->nmab", hessians, gradients
        ) + np.einsum("nka,nbkm->nmab", gradients, hessians)
        derivative_target = -0.5 * np.swapaxes(laplacian_gradients, 1, 2)
        right_hand_side = derivative_target - np.einsum(
            "nmab,b->nma", derivative_moment, coefficients
        )
        right_hand_side *= self.weights[:, None, None]
        _, diagnostics = self.reconstruct_parameters(trajectories)
        moment = diagnostics["moment"]
        scale = max(float(np.trace(moment)) / 3.0, 1.0)
        system = moment + self.regularization * scale * np.eye(3)
        coefficient_jacobian = np.linalg.solve(
            system, right_hand_side.reshape(-1, 3).T
        ).T.reshape(self.ntraj, self.nparticles, 3)
        return coefficient_jacobian / coefficients[None, None, :]

    def sample_initial(self, theta, *, warmup=500, proposal_scale=0.28):
        """Draw one sample from each independent random-walk Metropolis chain."""
        rng = np.random.default_rng(self.seed)
        coordinates = rng.normal(scale=0.7, size=(self.ntraj, self.nparticles))
        log_density = 2.0 * self.log_amplitude_terms(coordinates, theta)[0]
        accepted = 0
        for _ in range(int(warmup)):
            proposal = coordinates + rng.normal(
                scale=proposal_scale, size=coordinates.shape
            )
            proposal_log_density = 2.0 * self.log_amplitude_terms(proposal, theta)[0]
            accept = np.log(rng.random(self.ntraj)) < (
                proposal_log_density - log_density
            )
            coordinates[accept] = proposal[accept]
            log_density[accept] = proposal_log_density[accept]
            accepted += np.count_nonzero(accept)
        self.acceptance = accepted / (self.ntraj * warmup)
        return coordinates

    def _sample_state(self, theta, trajectories, tangent_data=None):
        _, _, _, _, observables = self.log_amplitude_terms(trajectories, theta)
        quantum_potential, quantum_force = self.quantum_potential_force(
            trajectories, theta
        )
        local_energy = self.potential(trajectories) + quantum_potential
        energy = float(np.dot(self.weights, local_energy))
        centered_observables = observables - np.einsum(
            "n,na->a", self.weights, observables
        )
        gradient = 2.0 * np.einsum(
            "n,na,n->a", self.weights, centered_observables, local_energy - energy
        )
        state = {
            "energy": energy,
            "gradient": gradient,
            "local_energy": local_energy,
            "variance": float(np.dot(self.weights, (local_energy - energy) ** 2)),
        }
        if tangent_data is None:
            return state
        tangents, metric, diagnostics = tangent_data
        residual_force = self.classical_force(trajectories) + quantum_force
        force = np.einsum(
            "n,nia,ni->a", self.weights, tangents, residual_force
        )
        state.update(
            tangents=tangents,
            metric=metric,
            force=force,
            residual_force=residual_force,
            tangent_diagnostics=diagnostics,
        )
        return state

    def constrained_continuity_lift(self, trajectories, theta=None):
        if theta is None:
            theta, reconstruction = self.reconstruct_parameters(trajectories)
        else:
            _, reconstruction = self.reconstruct_parameters(trajectories)
        jacobian = self.reconstruction_jacobian(trajectories, theta)
        values, gradients, labels = self.transport_basis.values_and_gradients(
            trajectories[:, :, None]
        )
        gradients = gradients[:, :, 0, :]
        _, _, _, _, observables = self.log_amplitude_terms(trajectories, theta)
        scores = 2.0 * (
            observables - np.einsum("n,na->a", self.weights, observables)
        )
        centered_values = values - np.einsum(
            "n,nk->k", self.weights, values
        )[None, :]
        weak_target = np.einsum(
            "n,nk,na->ka", self.weights, centered_values, scores
        )
        kinetic = np.einsum(
            "n,nik,nil->kl", self.weights, gradients / self.mass, gradients
        )
        eigenvalues, eigenvectors = np.linalg.eigh(kinetic)
        largest = max(float(eigenvalues[-1]), 1.0e-30)
        retained = eigenvalues > self.regularization * largest
        if np.count_nonzero(retained) < 4:
            retained[-4:] = True
        whitening = eigenvectors[:, retained] / np.sqrt(
            np.maximum(eigenvalues[retained], self.regularization * largest)
        )[None, :]
        whitened_gradients = np.einsum("nik,kl->nil", gradients, whitening)
        whitened_target = whitening.T @ weak_target
        whitened_kinetic = np.einsum(
            "n,nik,nil->kl",
            self.weights,
            whitened_gradients / self.mass,
            whitened_gradients,
        )
        reconstruction_constraint = np.einsum(
            "nia,nik->ak", jacobian, whitened_gradients / self.mass
        )
        sample_state = self._sample_state(theta, trajectories)
        _, quantum_force = self.quantum_potential_force(trajectories, theta)
        residual_force = self.classical_force(trajectories) + quantum_force
        force_constraint = np.einsum(
            "n,nik,ni->k",
            self.weights,
            whitened_gradients / self.mass,
            residual_force,
        )
        constraint = np.vstack(
            (reconstruction_constraint, force_constraint[None, :])
        )
        target = np.vstack((np.eye(3), -sample_state["gradient"][None, :]))
        basis_size = whitened_kinetic.shape[0]
        kkt = np.block(
            [
                [
                    whitened_kinetic
                    + self.regularization * np.eye(basis_size),
                    constraint.T,
                ],
                [constraint, np.zeros((4, 4))],
            ]
        )
        solution = np.linalg.lstsq(
            kkt,
            np.vstack((whitened_target, target)),
            rcond=self.regularization,
        )[0]
        coefficients = whitening @ solution[:basis_size]
        tangents = np.einsum(
            "nik,ka->nia", gradients / self.mass, coefficients
        )
        metric = self.mass * np.einsum(
            "n,nia,nib->ab", self.weights, tangents, tangents
        )
        diagnostics = dict(reconstruction)
        diagnostics.update(
            lift_identity=np.einsum("nia,nib->ab", jacobian, tangents),
            weak_residual=kinetic @ coefficients - weak_target,
            retained_basis_rank=int(np.count_nonzero(retained)),
            condition=float(np.linalg.cond(kkt)),
            labels=labels,
            jacobian=jacobian,
        )
        return theta, tangents, metric, diagnostics

    def _flow_rhs(self, trajectories):
        theta, tangents, metric, diagnostics = self.constrained_continuity_lift(
            trajectories
        )
        state = self._sample_state(
            theta, trajectories, (tangents, metric, diagnostics)
        )
        regularized_metric = metric + self.regularization * np.eye(3)
        parameter_velocity = np.linalg.solve(
            self.friction * regularized_metric, state["force"]
        )
        trajectory_velocity = np.einsum(
            "nia,a->ni", tangents, parameter_velocity
        )
        diagnostics["parameter_velocity"] = parameter_velocity
        diagnostics["kinematic_velocity"] = np.einsum(
            "nia,ni->a", diagnostics["jacobian"], trajectory_velocity
        )
        return trajectory_velocity, theta, parameter_velocity, state, diagnostics

    def _rk4_step(self, trajectories, step, first_velocity):
        k1 = first_velocity
        k2 = self._flow_rhs(trajectories + 0.5 * step * k1)[0]
        k3 = self._flow_rhs(trajectories + 0.5 * step * k2)[0]
        k4 = self._flow_rhs(trajectories + step * k3)[0]
        return trajectories + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _grid_state(self, theta):
        x1, x2, x3 = np.meshgrid(self.grid, self.grid, self.grid, indexing="ij")
        coordinates = np.stack((x1, x2, x3), axis=-1)
        amplitude, _, _, _, observables = self.log_amplitude_terms(
            coordinates, theta
        )
        density = np.exp(2.0 * (amplitude - np.max(amplitude)))
        dx = self.grid[1] - self.grid[0]
        density /= np.sum(density) * dx**3
        quantum_potential, _ = self.quantum_potential_force(coordinates, theta)
        local_energy = self.potential(coordinates) + quantum_potential
        energy = float(np.sum(density * local_energy) * dx**3)
        means = np.sum(
            density[..., None] * observables, axis=(0, 1, 2)
        ) * dx**3
        gradient = 2.0 * np.sum(
            density[..., None]
            * (observables - means)
            * (local_energy - energy)[..., None],
            axis=(0, 1, 2),
        ) * dx**3
        return density, energy, gradient

    def optimize_grid_jastrow(self, theta0=None):
        if theta0 is None:
            theta0 = np.log((1.2, 0.7, 0.35))
        result = minimize(
            lambda theta: self._grid_state(theta)[1],
            np.asarray(theta0),
            jac=lambda theta: self._grid_state(theta)[2],
            method="BFGS",
            options={"gtol": 1.0e-9, "maxiter": 100},
        )
        return result

    def run(
        self,
        *,
        theta0=(np.log(1.2), np.log(0.7), np.log(0.35)),
        dt=0.002,
        max_steps=300,
        tolerance=1.0e-2,
        max_parameter_step=0.006,
        record_every=5,
        warmup=500,
    ):
        trajectories = self.sample_initial(np.asarray(theta0), warmup=warmup)
        times, energies, parameters, gradients, forces, velocities = [], [], [], [], [], []
        time = 0.0
        for step_index in range(max_steps + 1):
            trajectory_velocity, theta, parameter_velocity, state, diagnostics = (
                self._flow_rhs(trajectories)
            )
            converged = (
                np.linalg.norm(parameter_velocity, ord=np.inf) < tolerance
                and np.linalg.norm(state["gradient"], ord=np.inf) < tolerance
            )
            if step_index % record_every == 0 or step_index == max_steps or converged:
                times.append(time)
                energies.append(state["energy"])
                parameters.append(theta.copy())
                gradients.append(state["gradient"].copy())
                forces.append(state["force"].copy())
                velocities.append(parameter_velocity.copy())
            if converged:
                self.success = True
                self.message = "three-particle constrained flow converged"
                break
            if step_index == max_steps:
                self.message = "maximum steps reached"
                break
            step = min(
                dt,
                max_parameter_step
                / max(np.linalg.norm(parameter_velocity, ord=np.inf), 1.0e-15),
            )
            trajectories = self._rk4_step(
                trajectories, step, trajectory_velocity
            )
            time += step
        _, theta, parameter_velocity, state, diagnostics = self._flow_rhs(
            trajectories
        )
        _, reference_energy, reference_gradient = self._grid_state(theta)
        self.theta = theta
        self.x = trajectories
        self.energy = state["energy"]
        self.gradient = state["gradient"]
        self.reference_energy = reference_energy
        self.reference_gradient = reference_gradient
        self.parameter_velocity = parameter_velocity
        self.force_gradient_gap = state["force"] + state["gradient"]
        self.lift_error = float(
            np.max(np.abs(diagnostics["lift_identity"] - np.eye(3)))
        )
        self.kinematic_error = diagnostics["kinematic_velocity"] - parameter_velocity
        self.weak_residual = diagnostics["weak_residual"]
        self.history = {
            "time": np.asarray(times),
            "energy": np.asarray(energies),
            "theta": np.asarray(parameters),
            "gradient": np.asarray(gradients),
            "force": np.asarray(forces),
            "parameter_velocity": np.asarray(velocities),
        }
        return self

