"""Invariant neural scalar potentials for equivariant trajectory transport."""

from __future__ import annotations

import numpy as np


class InvariantNeuralTransportPotential:
    r"""A shared one-/two-/three-body invariant neural potential.

    The network returns one scalar potential per density parameter. Cartesian
    differentiation makes the associated velocity field automatically
    permutation and orthogonal-transformation equivariant.
    """

    def __init__(
        self,
        nparameters,
        *,
        hidden_width=16,
        include_three_body=False,
        seed=0,
    ):
        if int(nparameters) < 1 or int(hidden_width) < 1:
            raise ValueError("nparameters and hidden_width must be positive")
        self.nparameters = int(nparameters)
        self.hidden_width = int(hidden_width)
        self.include_three_body = bool(include_three_body)
        self.seed = int(seed)
        self.parameters = None
        self.loss_history = None
        self.constrained_readout = None
        self.constraint_diagnostics = None

    @staticmethod
    def _jax():
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError(
                "InvariantNeuralTransportPotential requires JAX."
            ) from error
        jax.config.update("jax_enable_x64", True)
        return jax, jnp

    def _initialize(self):
        jax, jnp = self._jax()
        keys = iter(jax.random.split(jax.random.PRNGKey(self.seed), 6))
        width, output = self.hidden_width, self.nparameters

        def branch(input_size):
            scale1 = np.sqrt(2.0 / input_size)
            scale2 = np.sqrt(2.0 / width)
            return {
                "w1": scale1 * jax.random.normal(next(keys), (input_size, width)),
                "b1": jnp.zeros(width),
                "w2": scale2 * jax.random.normal(next(keys), (width, output)),
                "b2": jnp.zeros(output),
            }

        parameters = {"one": branch(1), "pair": branch(1)}
        if self.include_three_body:
            parameters["three"] = branch(3)
        self.parameters = parameters

    def _potential_function(self):
        jax, jnp = self._jax()
        include_three_body = self.include_three_body

        def mlp(parameters, inputs):
            return jnp.tanh(inputs @ parameters["w1"] + parameters["b1"]) @ parameters[
                "w2"
            ] + parameters["b2"]

        def potential(parameters, coordinates):
            particles = coordinates.shape[0]
            radii = jnp.sqrt(jnp.sum(coordinates**2, axis=1) + 1.0e-16)
            value = jnp.sum(mlp(parameters["one"], radii[:, None]), axis=0)
            pair_distances = {}
            for particle_i in range(particles):
                for particle_j in range(particle_i + 1, particles):
                    distance = jnp.sqrt(
                        jnp.sum(
                            (coordinates[particle_i] - coordinates[particle_j]) ** 2
                        )
                        + 1.0e-16
                    )
                    pair_distances[(particle_i, particle_j)] = distance
                    value = value + mlp(parameters["pair"], distance[None])
            if include_three_body:
                for particle_i in range(particles):
                    for particle_j in range(particle_i + 1, particles):
                        for particle_k in range(particle_j + 1, particles):
                            edges = jnp.sort(
                                jnp.asarray(
                                    (
                                        pair_distances[(particle_i, particle_j)],
                                        pair_distances[(particle_i, particle_k)],
                                        pair_distances[(particle_j, particle_k)],
                                    )
                                )
                            )
                            value = value + mlp(parameters["three"], edges)
            return value

        return potential

    def _feature_function(self):
        """Return invariant nonlinear features before the linear readout."""
        _, jnp = self._jax()
        include_three_body = self.include_three_body

        def hidden(parameters, inputs):
            return jnp.tanh(inputs @ parameters["w1"] + parameters["b1"])

        def features(parameters, coordinates):
            particles = coordinates.shape[0]
            radii = jnp.sqrt(jnp.sum(coordinates**2, axis=1) + 1.0e-16)
            branches = [
                jnp.sum(hidden(parameters["one"], radii[:, None]), axis=0)
            ]
            pair_distances = {}
            pair_features = jnp.zeros(self.hidden_width)
            for particle_i in range(particles):
                for particle_j in range(particle_i + 1, particles):
                    distance = jnp.sqrt(
                        jnp.sum(
                            (coordinates[particle_i] - coordinates[particle_j]) ** 2
                        )
                        + 1.0e-16
                    )
                    pair_distances[(particle_i, particle_j)] = distance
                    pair_features = pair_features + hidden(
                        parameters["pair"], distance[None]
                    )
            branches.append(pair_features)
            if include_three_body:
                three_features = jnp.zeros(self.hidden_width)
                for particle_i in range(particles):
                    for particle_j in range(particle_i + 1, particles):
                        for particle_k in range(particle_j + 1, particles):
                            edges = jnp.sort(
                                jnp.asarray(
                                    (
                                        pair_distances[(particle_i, particle_j)],
                                        pair_distances[(particle_i, particle_k)],
                                        pair_distances[(particle_j, particle_k)],
                                    )
                                )
                            )
                            three_features = three_features + hidden(
                                parameters["three"], edges
                            )
                branches.append(three_features)
            return jnp.concatenate(branches)

        return features

    def feature_values_and_gradients(self, coordinates, parameters=None):
        """Evaluate invariant hidden features and equivariant gradients."""
        jax, jnp = self._jax()
        if parameters is None:
            if self.parameters is None:
                self._initialize()
            parameters = self.parameters
        features = self._feature_function()
        coordinates = jnp.asarray(coordinates, dtype=float)
        values = jax.vmap(features, in_axes=(None, 0))(parameters, coordinates)
        jacobian = jax.vmap(jax.jacrev(features, argnums=1), in_axes=(None, 0))(
            parameters, coordinates
        )
        gradients = jnp.transpose(jacobian, (0, 2, 3, 1))
        return np.asarray(values), np.asarray(gradients)

    def fit_constrained_readout(
        self,
        coordinates,
        scores,
        *,
        jacobian,
        residual_force,
        energy_gradient,
        weights=None,
        mass=1.0,
        regularization=1.0e-9,
    ):
        r"""Fit the linear neural readout with exact particle constraints."""
        values, gradients = self.feature_values_and_gradients(coordinates)
        scores = np.asarray(scores, dtype=float)
        jacobian = np.asarray(jacobian, dtype=float)
        residual_force = np.asarray(residual_force, dtype=float)
        energy_gradient = np.asarray(energy_gradient, dtype=float)
        samples = values.shape[0]
        if weights is None:
            weights = np.full(samples, 1.0 / samples)
        weights = np.asarray(weights, dtype=float)
        centered = values - np.einsum("n,nk->k", weights, values)[None, :]
        kinetic = np.einsum(
            "n,npdk,npdl->kl", weights, gradients / mass, gradients
        )
        weak_target = np.einsum(
            "n,nk,na->ka", weights, centered, scores
        )
        scales = np.sqrt(np.maximum(np.diag(kinetic), 1.0e-24))
        scaled_gradients = gradients / scales[None, None, None, :]
        scaled_kinetic = np.einsum(
            "n,npdk,npdl->kl",
            weights,
            scaled_gradients / mass,
            scaled_gradients,
        )
        scaled_target = weak_target / scales[:, None]
        reconstruction_constraint = np.einsum(
            "npda,npdk->ak", jacobian, scaled_gradients / mass
        )
        force_constraint = np.einsum(
            "n,npdk,npd->k",
            weights,
            scaled_gradients / mass,
            residual_force,
        )
        constraint = np.vstack(
            (reconstruction_constraint, force_constraint[None, :])
        )
        target = np.vstack(
            (np.eye(self.nparameters), -energy_gradient[None, :])
        )
        feature_count = scaled_kinetic.shape[0]
        constraint_count = constraint.shape[0]
        kkt = np.block(
            [
                [
                    scaled_kinetic + regularization * np.eye(feature_count),
                    constraint.T,
                ],
                [constraint, np.zeros((constraint_count, constraint_count))],
            ]
        )
        solution = np.linalg.lstsq(
            kkt, np.vstack((scaled_target, target)), rcond=regularization
        )[0]
        scaled_readout = solution[:feature_count]
        self.constrained_readout = scaled_readout / scales[:, None]
        potentials = values @ self.constrained_readout
        tangents = np.einsum(
            "npdk,ka->npda", gradients / mass, self.constrained_readout
        )
        lift = np.einsum("npda,npdb->ab", jacobian, tangents)
        projected_force = np.einsum(
            "n,npda,npd->a", weights, tangents, residual_force
        )
        self.constraint_diagnostics = {
            "lift_identity": lift,
            "force_gradient_gap": projected_force + energy_gradient,
            "condition": float(np.linalg.cond(kkt)),
            "potentials": potentials,
            "tangents": tangents,
        }
        return self

    def constrained_values_and_gradients(self, coordinates):
        """Evaluate the hard-constrained learned scalar potential."""
        if self.constrained_readout is None:
            raise RuntimeError("call fit_constrained_readout first")
        values, gradients = self.feature_values_and_gradients(coordinates)
        return (
            values @ self.constrained_readout,
            np.einsum("npdk,ka->npda", gradients, self.constrained_readout),
        )

    def values_and_gradients(self, coordinates, parameters=None):
        """Return neural potentials and coordinate gradients."""
        jax, jnp = self._jax()
        if parameters is None:
            if self.parameters is None:
                self._initialize()
            parameters = self.parameters
        potential = self._potential_function()
        coordinates = jnp.asarray(coordinates, dtype=float)
        values = jax.vmap(potential, in_axes=(None, 0))(parameters, coordinates)
        jacobian = jax.vmap(jax.jacrev(potential, argnums=1), in_axes=(None, 0))(
            parameters, coordinates
        )
        gradients = jnp.transpose(jacobian, (0, 2, 3, 1))
        return np.asarray(values), np.asarray(gradients)

    def fit(
        self,
        coordinates,
        scores,
        *,
        weights=None,
        mass=1.0,
        jacobian=None,
        residual_force=None,
        energy_gradient=None,
        constraint_penalty=100.0,
        learning_rate=2.0e-3,
        steps=200,
    ):
        r"""Train with the constrained weak-Poisson objective.

        Optional reconstruction and energy constraints penalize $JU-I$ and
        $\mathcal F+\nabla E$ using exactly the same particle estimators as the
        fixed-basis KKT formulation.
        """
        jax, jnp = self._jax()
        if self.parameters is None:
            self._initialize()
        coordinates = jnp.asarray(coordinates, dtype=float)
        scores = jnp.asarray(scores, dtype=float)
        samples = coordinates.shape[0]
        if scores.shape != (samples, self.nparameters):
            raise ValueError("scores must have shape (samples, nparameters)")
        if weights is None:
            weights = np.full(samples, 1.0 / samples)
        weights = jnp.asarray(weights, dtype=float)
        potential = self._potential_function()
        batch_potential = jax.vmap(potential, in_axes=(None, 0))
        batch_jacobian = jax.vmap(
            jax.jacrev(potential, argnums=1), in_axes=(None, 0)
        )
        identity = jnp.eye(self.nparameters)
        empirical_jacobian = None if jacobian is None else jnp.asarray(jacobian)
        force = None if residual_force is None else jnp.asarray(residual_force)
        gradient_target = (
            None if energy_gradient is None else jnp.asarray(energy_gradient)
        )

        def loss(parameters):
            values = batch_potential(parameters, coordinates)
            gradients = jnp.transpose(
                batch_jacobian(parameters, coordinates), (0, 2, 3, 1)
            )
            centered = values - jnp.einsum("n,na->a", weights, values)[None, :]
            weak = 0.5 * jnp.einsum(
                "n,npda,npda->", weights, gradients / mass, gradients
            ) - jnp.einsum("n,na,na->", weights, centered, scores)
            penalty = 0.0
            if empirical_jacobian is not None:
                lift = jnp.einsum(
                    "npda,npdb->ab", empirical_jacobian, gradients / mass
                )
                penalty = penalty + jnp.sum((lift - identity) ** 2)
            if force is not None and gradient_target is not None:
                projected = jnp.einsum(
                    "n,npda,npd->a", weights, gradients / mass, force
                )
                penalty = penalty + jnp.sum((projected + gradient_target) ** 2)
            return weak + constraint_penalty * penalty

        value_and_gradient = jax.jit(jax.value_and_grad(loss))
        parameters = self.parameters
        first_moment = jax.tree.map(jnp.zeros_like, parameters)
        second_moment = jax.tree.map(jnp.zeros_like, parameters)
        history = []
        for step_index in range(1, int(steps) + 1):
            value, gradient = value_and_gradient(parameters)
            first_moment = jax.tree.map(
                lambda moment, derivative: 0.9 * moment + 0.1 * derivative,
                first_moment,
                gradient,
            )
            second_moment = jax.tree.map(
                lambda moment, derivative: 0.999 * moment + 0.001 * derivative**2,
                second_moment,
                gradient,
            )
            corrected_first = jax.tree.map(
                lambda moment: moment / (1.0 - 0.9**step_index), first_moment
            )
            corrected_second = jax.tree.map(
                lambda moment: moment / (1.0 - 0.999**step_index), second_moment
            )
            parameters = jax.tree.map(
                lambda parameter, first, second: parameter
                - learning_rate * first / (jnp.sqrt(second) + 1.0e-8),
                parameters,
                corrected_first,
                corrected_second,
            )
            history.append(float(value))
        self.parameters = parameters
        self.loss_history = np.asarray(history)
        return self
