"""Particle-fitted amplitude corrections for quantum-force closure."""

from __future__ import annotations

import numpy as np


def global_polynomial_jastrow_terms(
    coordinates, parameters, *, pair_width=0.8
):
    r"""Return $A$, $\nabla A$, and $\nabla^2A$ for a global baseline.

    ``parameters = (a, log_b, d, log_c)`` defines

    $$
    A=-\frac a2\sum_i x_i^2-\frac b4\sum_i x_i^4+d\sum_i x_i
      -c\sum_{i<j}e^{-(x_i-x_j)^2/(2\sigma^2)}.
    $$
    """
    coordinates = np.asarray(coordinates, dtype=float)
    a, log_b, displacement, log_c = np.asarray(parameters, dtype=float)
    b, c = np.exp(log_b), np.exp(log_c)
    amplitude = (
        -0.5 * a * np.sum(coordinates**2, axis=-1)
        - 0.25 * b * np.sum(coordinates**4, axis=-1)
        + displacement * np.sum(coordinates, axis=-1)
    )
    gradient = -a * coordinates - b * coordinates**3 + displacement
    laplacian = np.sum(-a - 3.0 * b * coordinates**2, axis=-1)
    sigma = float(pair_width)
    particles = coordinates.shape[-1]
    for first in range(particles):
        for second in range(first + 1, particles):
            separation = coordinates[..., first] - coordinates[..., second]
            gaussian = np.exp(-0.5 * (separation / sigma) ** 2)
            amplitude -= c * gaussian
            derivative = c * separation * gaussian / sigma**2
            second_derivative = c * gaussian * (
                1.0 / sigma**2 - separation**2 / sigma**4
            )
            gradient[..., first] += derivative
            gradient[..., second] -= derivative
            laplacian += 2.0 * second_derivative
    return amplitude, gradient, laplacian


class SharedLinearScoreCorrection1D:
    r"""Shared one-/two-/three-body linear amplitude correction in 1D."""

    def __init__(
        self,
        *,
        one_body_centers=(-1.5, -0.75, 0.0, 0.75, 1.5),
        one_body_width=1.3,
        pair_widths=(1.0, 1.6),
        three_body_scales=(0.18,),
        regularization=1.0e-3,
    ):
        self.one_body_centers = np.asarray(one_body_centers, dtype=float)
        self.one_body_width = float(one_body_width)
        self.pair_widths = np.asarray(pair_widths, dtype=float)
        self.three_body_scales = np.asarray(three_body_scales, dtype=float)
        self.regularization = float(regularization)
        self.coefficients = None
        self.diagnostics = None
        self._quantum_force_kernel = None

    @property
    def size(self):
        return (
            self.one_body_centers.size
            + self.pair_widths.size
            + self.three_body_scales.size
        )

    def basis_terms(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        samples, particles = coordinates.shape
        values, gradients, laplacians = [], [], []

        width = self.one_body_width
        displacement = (
            coordinates[:, :, None] - self.one_body_centers[None, None, :]
        )
        one_values_per_particle = np.exp(-0.5 * (displacement / width) ** 2)
        one_gradients = -(displacement / width**2) * one_values_per_particle
        one_laplacians = (
            displacement**2 / width**4 - 1.0 / width**2
        ) * one_values_per_particle
        values.append(np.sum(one_values_per_particle, axis=1))
        gradients.append(one_gradients)
        laplacians.append(np.sum(one_laplacians, axis=1))

        pair_values = np.zeros((samples, self.pair_widths.size))
        pair_gradients = np.zeros((samples, particles, self.pair_widths.size))
        pair_laplacians = np.zeros((samples, self.pair_widths.size))
        pair_square_sum = np.zeros(samples)
        pair_square_gradient = np.zeros((samples, particles))
        for first in range(particles):
            for second in range(first + 1, particles):
                separation = coordinates[:, first] - coordinates[:, second]
                square = separation**2
                pair_square_sum += square
                pair_square_gradient[:, first] += 2.0 * separation
                pair_square_gradient[:, second] -= 2.0 * separation
                gaussian = np.exp(
                    -0.5 * square[:, None] / self.pair_widths[None, :] ** 2
                )
                derivative = (
                    -separation[:, None]
                    / self.pair_widths[None, :] ** 2
                    * gaussian
                )
                pair_values += gaussian
                pair_gradients[:, first] += derivative
                pair_gradients[:, second] -= derivative
                pair_laplacians += 2.0 * gaussian * (
                    square[:, None] / self.pair_widths[None, :] ** 4
                    - 1.0 / self.pair_widths[None, :] ** 2
                )
        values.append(pair_values)
        gradients.append(pair_gradients)
        laplacians.append(pair_laplacians)

        if particles >= 3 and self.three_body_scales.size:
            three_values = np.exp(
                -pair_square_sum[:, None] * self.three_body_scales[None, :]
            )
            three_gradients = (
                -pair_square_gradient[:, :, None]
                * self.three_body_scales[None, None, :]
                * three_values[:, None, :]
            )
            second_trace = 2.0 * particles * (particles - 1)
            gradient_square = np.sum(pair_square_gradient**2, axis=1)
            three_laplacians = three_values * (
                self.three_body_scales[None, :] ** 2
                * gradient_square[:, None]
                - self.three_body_scales[None, :] * second_trace
            )
            values.append(three_values)
            gradients.append(three_gradients)
            laplacians.append(three_laplacians)

        return (
            np.concatenate(values, axis=1),
            np.concatenate(gradients, axis=2),
            np.concatenate(laplacians, axis=1),
        )

    def fit(self, coordinates, baseline_gradient, weights=None):
        """Fit the amplitude score correction by linear score matching."""
        coordinates = np.asarray(coordinates, dtype=float)
        baseline_gradient = np.asarray(baseline_gradient, dtype=float)
        values, gradients, laplacians = self.basis_terms(coordinates)
        samples = coordinates.shape[0]
        if weights is None:
            weights = np.full(samples, 1.0 / samples)
        weights = np.asarray(weights, dtype=float)
        gram = np.einsum("n,nik,nil->kl", weights, gradients, gradients)
        target = -0.5 * np.einsum("n,nk->k", weights, laplacians) - np.einsum(
            "n,nik,ni->k", weights, gradients, baseline_gradient
        )
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        largest = max(float(eigenvalues[-1]), 1.0e-30)
        ridge = self.regularization * largest
        inverse = 1.0 / (eigenvalues + ridge)
        coefficients = eigenvectors @ (inverse * (eigenvectors.T @ target))
        self.coefficients = coefficients
        self.diagnostics = {
            "normal_residual": gram @ coefficients - target,
            "rank": int(np.count_nonzero(eigenvalues > ridge)),
            "condition": float(
                largest / max(eigenvalues[eigenvalues > self.regularization * largest][0], 1e-30)
            ),
        }
        return self

    def terms(self, coordinates):
        if self.coefficients is None:
            raise RuntimeError("fit the correction before evaluation")
        values, gradients, laplacians = self.basis_terms(coordinates)
        return (
            values @ self.coefficients,
            np.einsum("nik,k->ni", gradients, self.coefficients),
            laplacians @ self.coefficients,
        )

    def quantum_potential_force(
        self,
        coordinates,
        baseline_parameters,
        *,
        pair_width=0.8,
        mass=1.0,
        hbar=1.0,
    ):
        """Evaluate the fitted quantum potential and force with AD."""
        if self.coefficients is None:
            raise RuntimeError("fit the correction before evaluating its force")
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError("the AD quantum force requires JAX") from error
        jax.config.update("jax_enable_x64", True)
        centers = jnp.asarray(self.one_body_centers)
        pair_widths = jnp.asarray(self.pair_widths)
        three_scales = jnp.asarray(self.three_body_scales)
        width = self.one_body_width
        def amplitude(coefficients, baseline_parameters, point):
            a, log_b, displacement, log_c = baseline_parameters
            b, c = jnp.exp(log_b), jnp.exp(log_c)
            value = (
                -0.5 * a * jnp.sum(point**2)
                - 0.25 * b * jnp.sum(point**4)
                + displacement * jnp.sum(point)
            )
            one = jnp.sum(
                jnp.exp(-0.5 * ((point[:, None] - centers[None, :]) / width) ** 2),
                axis=0,
            )
            pair_features = jnp.zeros(pair_widths.size)
            pair_square_sum = 0.0
            for first in range(point.size):
                for second in range(first + 1, point.size):
                    separation = point[first] - point[second]
                    square = separation**2
                    pair_square_sum = pair_square_sum + square
                    baseline_pair = jnp.exp(-0.5 * (separation / pair_width) ** 2)
                    value = value - c * baseline_pair
                    pair_features = pair_features + jnp.exp(
                        -0.5 * square / pair_widths**2
                    )
            features = [one, pair_features]
            if three_scales.size:
                features.append(jnp.exp(-pair_square_sum * three_scales))
            return value + jnp.concatenate(features) @ coefficients

        amplitude_gradient = jax.grad(amplitude, argnums=2)
        amplitude_hessian = jax.hessian(amplitude, argnums=2)

        def quantum_potential(coefficients, baseline_parameters, point):
            gradient = amplitude_gradient(coefficients, baseline_parameters, point)
            hessian = amplitude_hessian(coefficients, baseline_parameters, point)
            return -(hbar**2 / (2.0 * mass)) * (
                jnp.trace(hessian) + jnp.dot(gradient, gradient)
            )

        if self._quantum_force_kernel is None:
            self._quantum_force_kernel = jax.jit(
                jax.vmap(
                    jax.value_and_grad(quantum_potential, argnums=2),
                    in_axes=(None, None, 0),
                )
            )
        potential, gradient = self._quantum_force_kernel(
            jnp.asarray(self.coefficients),
            jnp.asarray(baseline_parameters),
            jnp.asarray(coordinates, dtype=float),
        )
        return np.asarray(potential), -np.asarray(gradient)


class InvariantNeuralScoreCorrection1D:
    r"""Smooth invariant neural amplitude correction for identical particles."""

    def __init__(self, *, hidden_width=16, seed=0, pair_width=0.8):
        self.hidden_width = int(hidden_width)
        self.seed = int(seed)
        self.pair_width = float(pair_width)
        self.parameters = None
        self.loss_history = None
        self._quantum_force_kernel = None

    @staticmethod
    def _jax():
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError("neural score correction requires JAX") from error
        jax.config.update("jax_enable_x64", True)
        return jax, jnp

    def _initialize(self):
        jax, jnp = self._jax()
        keys = iter(jax.random.split(jax.random.PRNGKey(self.seed), 8))
        width = self.hidden_width

        def branch(input_size):
            return {
                "w1": jax.random.normal(next(keys), (input_size, width))
                / np.sqrt(input_size),
                "b1": jnp.zeros(width),
                # Start exactly from the physical Jastrow baseline. Random
                # output weights can have harmless amplitude but enormous
                # third derivatives and therefore an unusable quantum force.
                "w2": jnp.zeros(width),
            }

        self.parameters = {
            "one": branch(1),
            "pair": branch(1),
            "three": branch(3),
        }

    def _functions(self, baseline_parameters):
        jax, jnp = self._jax()
        pair_width = self.pair_width
        baseline_parameters = jnp.asarray(baseline_parameters)

        def branch(parameters, inputs):
            hidden = jnp.tanh(inputs @ parameters["w1"] + parameters["b1"])
            return hidden @ parameters["w2"]

        def correction(parameters, coordinates):
            particles = coordinates.size
            value = jnp.sum(
                jax.vmap(lambda x: branch(parameters["one"], x[None]))(
                    coordinates
                )
            )
            pair_squares = []
            for first in range(particles):
                for second in range(first + 1, particles):
                    square = (coordinates[first] - coordinates[second]) ** 2
                    pair_squares.append(square)
                    value = value + branch(parameters["pair"], square[None])
            if particles >= 3:
                squares = jnp.asarray(pair_squares)
                invariants = jnp.asarray(
                    (jnp.sum(squares), jnp.sum(squares**2), jnp.prod(squares))
                )
                value = value + branch(parameters["three"], invariants)
            return value

        def baseline(coordinates):
            a, log_b, displacement, log_c = baseline_parameters
            b, c = jnp.exp(log_b), jnp.exp(log_c)
            value = (
                -0.5 * a * jnp.sum(coordinates**2)
                - 0.25 * b * jnp.sum(coordinates**4)
                + displacement * jnp.sum(coordinates)
            )
            for first in range(coordinates.size):
                for second in range(first + 1, coordinates.size):
                    separation = coordinates[first] - coordinates[second]
                    value = value - c * jnp.exp(
                        -0.5 * (separation / pair_width) ** 2
                    )
            return value

        def total(parameters, coordinates):
            return baseline(coordinates) + correction(parameters, coordinates)

        return correction, total

    def fit(
        self,
        coordinates,
        baseline_parameters,
        *,
        steps=500,
        learning_rate=1.0e-3,
        correction_regularization=2.0e-4,
        force_smoothness=1.0e-3,
        validation_fraction=0.2,
        validation_every=10,
    ):
        """Fit the correction with exact-Hessian amplitude score matching."""
        jax, jnp = self._jax()
        if self.parameters is None:
            self._initialize()
        coordinates = jnp.asarray(coordinates, dtype=float)
        split = max(1, int((1.0 - validation_fraction) * coordinates.shape[0]))
        training_coordinates = coordinates[:split]
        validation_coordinates = coordinates[split:]
        correction, total = self._functions(baseline_parameters)
        total_gradient = jax.grad(total, argnums=1)
        total_hessian = jax.hessian(total, argnums=1)
        correction_gradient = jax.grad(correction, argnums=1)

        def correction_quantum_potential(parameters, point):
            gradient = correction_gradient(parameters, point)
            hessian = jax.hessian(correction, argnums=1)(parameters, point)
            return -(jnp.trace(hessian) + jnp.dot(gradient, gradient))

        correction_quantum_force = jax.grad(
            correction_quantum_potential, argnums=1
        )

        def point_loss(parameters, point):
            gradient = total_gradient(parameters, point)
            laplacian = jnp.trace(total_hessian(parameters, point))
            correction_norm = jnp.sum(correction_gradient(parameters, point) ** 2)
            force_norm = jnp.sum(correction_quantum_force(parameters, point) ** 2)
            return (
                jnp.sum(gradient**2)
                + laplacian
                + correction_regularization * correction_norm
                + force_smoothness * force_norm
            )

        def sample_loss(parameters, sample):
            return jnp.mean(jax.vmap(point_loss, in_axes=(None, 0))(parameters, sample))

        def loss(parameters):
            return sample_loss(parameters, training_coordinates)

        value_and_gradient = jax.jit(jax.value_and_grad(loss))
        parameters = self.parameters
        first = jax.tree.map(jnp.zeros_like, parameters)
        second = jax.tree.map(jnp.zeros_like, parameters)
        history = []
        validation_history = []
        best_parameters = parameters
        best_validation = float("inf")
        for step_index in range(1, int(steps) + 1):
            value, derivative = value_and_gradient(parameters)
            first = jax.tree.map(
                lambda moment, grad: 0.9 * moment + 0.1 * grad,
                first,
                derivative,
            )
            second = jax.tree.map(
                lambda moment, grad: 0.999 * moment + 0.001 * grad**2,
                second,
                derivative,
            )
            first_corrected = jax.tree.map(
                lambda moment: moment / (1.0 - 0.9**step_index), first
            )
            second_corrected = jax.tree.map(
                lambda moment: moment / (1.0 - 0.999**step_index), second
            )
            parameters = jax.tree.map(
                lambda parameter, m1, m2: parameter
                - learning_rate * m1 / (jnp.sqrt(m2) + 1.0e-8),
                parameters,
                first_corrected,
                second_corrected,
            )
            history.append(float(value))
            if step_index % int(validation_every) == 0 or step_index == int(steps):
                validation_value = float(
                    sample_loss(parameters, validation_coordinates)
                    if validation_coordinates.shape[0]
                    else value
                )
                validation_history.append((step_index, validation_value))
                if validation_value < best_validation:
                    best_validation = validation_value
                    best_parameters = jax.tree.map(lambda value: value.copy(), parameters)
        self.parameters = best_parameters
        self.loss_history = np.asarray(history)
        self.validation_history = np.asarray(validation_history)
        return self

    def terms(self, coordinates, baseline_parameters, *, batch_size=4096):
        """Evaluate correction amplitude, gradient, and Laplacian in batches."""
        if self.parameters is None:
            raise RuntimeError("fit the correction before evaluation")
        jax, jnp = self._jax()
        correction, _ = self._functions(baseline_parameters)
        gradient = jax.grad(correction, argnums=1)
        hessian = jax.hessian(correction, argnums=1)

        @jax.jit
        def evaluate(batch):
            values = jax.vmap(correction, in_axes=(None, 0))(self.parameters, batch)
            gradients = jax.vmap(gradient, in_axes=(None, 0))(self.parameters, batch)
            hessians = jax.vmap(hessian, in_axes=(None, 0))(self.parameters, batch)
            return values, gradients, jnp.trace(hessians, axis1=1, axis2=2)

        coordinates = np.asarray(coordinates, dtype=float)
        values, gradients, laplacians = [], [], []
        for start in range(0, coordinates.shape[0], int(batch_size)):
            result = evaluate(jnp.asarray(coordinates[start : start + batch_size]))
            values.append(np.asarray(result[0]))
            gradients.append(np.asarray(result[1]))
            laplacians.append(np.asarray(result[2]))
        return (
            np.concatenate(values),
            np.concatenate(gradients),
            np.concatenate(laplacians),
        )

    def quantum_potential_force(
        self,
        coordinates,
        baseline_parameters,
        *,
        mass=1.0,
        hbar=1.0,
        batch_size=2048,
    ):
        """Evaluate the learned quantum potential and force with AD."""
        if self.parameters is None:
            raise RuntimeError("fit the correction before evaluating its force")
        jax, jnp = self._jax()
        _, total = self._functions(baseline_parameters)
        amplitude_gradient = jax.grad(total, argnums=1)
        amplitude_hessian = jax.hessian(total, argnums=1)

        def quantum_potential(parameters, point):
            gradient = amplitude_gradient(parameters, point)
            hessian = amplitude_hessian(parameters, point)
            return -(hbar**2 / (2.0 * mass)) * (
                jnp.trace(hessian) + jnp.dot(gradient, gradient)
            )

        if self._quantum_force_kernel is None:
            self._quantum_force_kernel = jax.jit(
                jax.vmap(
                    jax.value_and_grad(quantum_potential, argnums=1),
                    in_axes=(None, 0),
                )
            )
        coordinates = np.asarray(coordinates, dtype=float)
        potentials, forces = [], []
        for start in range(0, coordinates.shape[0], int(batch_size)):
            potential, gradient = self._quantum_force_kernel(
                self.parameters,
                jnp.asarray(coordinates[start : start + batch_size]),
            )
            potentials.append(np.asarray(potential))
            forces.append(-np.asarray(gradient))
        return np.concatenate(potentials), np.concatenate(forces)
