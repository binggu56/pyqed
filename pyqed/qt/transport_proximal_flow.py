"""Jacobian-consistent proximal transport of a positive Jastrow density."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .direct_score_flow import DirectOverdampedScoreFlow1D
from .score_corrections import SharedLinearScoreCorrection1D


class JacobianProximalFlow1D(DirectOverdampedScoreFlow1D):
    r"""Optimize a composed density transport with the weak quantum energy.

    If $R=T(Z)$ and $Z\sim\rho_0$, the transported density is represented
    exactly by

    $$
    \rho_T(T(Z))=\rho_0(Z)/|\det\nabla_ZT|.
    $$

    The amplitude score is obtained from this change of variables, so the
    weak kinetic energy is the energy of a normalized density rather than an
    unconstrained score fit.  Only map derivatives through second order are
    needed; no pointwise quantum force is formed.
    """

    def __init__(self, *args, transport_model=None, training_fraction=0.75, **kwargs):
        kwargs["closure"] = "baseline"
        super().__init__(*args, **kwargs)
        self.transport_model = transport_model or SharedLinearScoreCorrection1D(
            one_body_centers=(-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8),
            one_body_width=1.0,
            pair_widths=(0.8, 1.2, 1.8),
            three_body_scales=(0.12, 0.3),
        )
        self.training_fraction = float(training_fraction)
        if not 0.0 < self.training_fraction < 1.0:
            raise ValueError("training_fraction must lie between zero and one")
        self.transport_whitenings = []
        self.transport_coefficients = []
        self.base_x = None
        self.diagnostics = None

    @staticmethod
    def _jax():
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError("Jacobian proximal flow requires JAX") from error
        jax.config.update("jax_enable_x64", True)
        return jax, jnp

    def _functions(self, whitenings=None, coefficients=None):
        jax, jnp = self._jax()
        model = self.transport_model
        centers = jnp.asarray(model.one_body_centers)
        one_width = model.one_body_width
        pair_widths = jnp.asarray(model.pair_widths)
        three_scales = jnp.asarray(model.three_body_scales)
        baseline = jnp.asarray(self.baseline_parameters)
        pair_width = self.pair_width

        if whitenings is None:
            whitenings = self.transport_whitenings
        if coefficients is None:
            coefficients = self.transport_coefficients
        whitenings = tuple(jnp.asarray(value) for value in whitenings)
        coefficients = tuple(jnp.asarray(value) for value in coefficients)

        def base_amplitude(point):
            a, log_b, displacement, log_c = baseline
            b, c = jnp.exp(log_b), jnp.exp(log_c)
            value = (
                -0.5 * a * jnp.sum(point**2)
                - 0.25 * b * jnp.sum(point**4)
                + displacement * jnp.sum(point)
            )
            for first in range(point.size):
                for second in range(first + 1, point.size):
                    separation = point[first] - point[second]
                    value = value - c * jnp.exp(
                        -0.5 * (separation / pair_width) ** 2
                    )
            return value

        def feature_values(point):
            one = jnp.sum(
                jnp.exp(
                    -0.5
                    * ((point[:, None] - centers[None, :]) / one_width) ** 2
                ),
                axis=0,
            )
            pair = jnp.zeros(pair_widths.size)
            pair_square_sum = 0.0
            for first in range(point.size):
                for second in range(first + 1, point.size):
                    separation = point[first] - point[second]
                    square = separation**2
                    pair_square_sum = pair_square_sum + square
                    pair = pair + jnp.exp(-0.5 * square / pair_widths**2)
            values = [one, pair]
            if three_scales.size:
                values.append(jnp.exp(-pair_square_sum * three_scales))
            return jnp.concatenate(values)

        feature_gradient = jax.jacfwd(feature_values)

        def apply_increment(point, whitening, coefficient):
            fields = feature_gradient(point).T @ whitening
            return point + fields @ coefficient

        def map_point(point):
            mapped = point
            for whitening, coefficient in zip(whitenings, coefficients):
                mapped = apply_increment(mapped, whitening, coefficient)
            return mapped

        map_jacobian = jax.jacfwd(map_point)

        def mapped_amplitude(point):
            jacobian = map_jacobian(point)
            _, log_determinant = jnp.linalg.slogdet(jacobian)
            return base_amplitude(point) - 0.5 * log_determinant

        mapped_amplitude_gradient = jax.grad(mapped_amplitude)

        def sample_state(point):
            mapped = map_point(point)
            jacobian = map_jacobian(point)
            determinant = jnp.linalg.det(jacobian)
            gradient_z = mapped_amplitude_gradient(point)
            score = jnp.linalg.solve(jacobian.T, gradient_z)
            one_body = self.barrier * (mapped**2 - self.well**2) ** 2
            one_body = one_body + self.tilt * mapped
            potential = jnp.sum(one_body)
            for first in range(mapped.size):
                for second in range(first + 1, mapped.size):
                    separation = mapped[first] - mapped[second]
                    potential = potential + self.interaction / jnp.sqrt(
                        separation**2 + self.softening**2
                    )
            energy = potential + self.hbar**2 / (2.0 * self.mass) * jnp.dot(
                score, score
            )
            return mapped, energy, determinant

        return jax, jnp, map_point, sample_state, apply_increment

    def state(self, base_coordinates, *, indices=None):
        """Evaluate transported weak energy on selected fixed base labels."""
        base_coordinates = np.asarray(base_coordinates, dtype=float)
        if indices is not None:
            base_coordinates = base_coordinates[np.asarray(indices)]
        jax, jnp, _, sample_state, _ = self._functions()
        mapped, energies, determinants = jax.jit(jax.vmap(sample_state))(
            jnp.asarray(base_coordinates)
        )
        mapped = np.asarray(mapped)
        energies = np.asarray(energies)
        determinants = np.asarray(determinants)
        energy = float(np.mean(energies))
        return {
            "x": mapped,
            "energy": energy,
            "variance": float(np.var(energies)),
            "standard_error": float(np.sqrt(np.var(energies) / energies.size)),
            "minimum_determinant": float(np.min(determinants)),
            "maximum_determinant": float(np.max(determinants)),
        }

    def quadrature_state(self, *, ngrid=31, xmax=3.7, batch_size=4096):
        """Independently audit the normalized transported density on a 3D grid."""
        grid = np.linspace(-float(xmax), float(xmax), int(ngrid))
        x1, x2, x3 = np.meshgrid(grid, grid, grid, indexing="ij")
        base_coordinates = np.stack((x1, x2, x3), axis=-1).reshape((-1, 3))
        amplitude = self.baseline_terms(base_coordinates)[0]
        weights = np.exp(2.0 * (amplitude - np.max(amplitude)))
        weights /= np.sum(weights)
        jax, jnp, _, sample_state, _ = self._functions()
        kernel = jax.jit(jax.vmap(sample_state))
        energy_sum = 0.0
        energy_square_sum = 0.0
        minimum_determinant = np.inf
        maximum_determinant = -np.inf
        for start in range(0, base_coordinates.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), base_coordinates.shape[0])
            _, energies, determinants = kernel(jnp.asarray(base_coordinates[start:stop]))
            energies = np.asarray(energies)
            determinants = np.asarray(determinants)
            batch_weights = weights[start:stop]
            energy_sum += float(np.dot(batch_weights, energies))
            energy_square_sum += float(np.dot(batch_weights, energies**2))
            minimum_determinant = min(minimum_determinant, float(np.min(determinants)))
            maximum_determinant = max(maximum_determinant, float(np.max(determinants)))
        return {
            "energy": energy_sum,
            "variance": max(energy_square_sum - energy_sum**2, 0.0),
            "minimum_determinant": minimum_determinant,
            "maximum_determinant": maximum_determinant,
            "ngrid": int(ngrid),
        }

    def _whitening(self, coordinates, training_indices):
        _, gradients, _ = self.transport_model.basis_terms(coordinates)
        gradients = gradients[np.asarray(training_indices)]
        metric = np.einsum("nik,nil->kl", gradients, gradients) / gradients.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        largest = max(float(eigenvalues[-1]), 1.0e-30)
        retained = eigenvalues > 1.0e-7 * largest
        whitening = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])[None, :]
        return whitening, eigenvalues

    def proximal_step(
        self,
        *,
        time_step=0.08,
        max_coefficient=0.06,
        maximum_iterations=40,
    ):
        """Append one map chosen using only the fixed training labels."""
        if self.base_x is None:
            raise RuntimeError("initialize the base cloud before taking a step")
        split = int(self.training_fraction * self.ntraj)
        training = np.arange(split)
        current = self.state(self.base_x)
        whitening, eigenvalues = self._whitening(current["x"], training)
        current_training = self.state(self.base_x, indices=training)
        jax, jnp, _, _, apply_increment = self._functions()
        base_training = jnp.asarray(self.base_x[training])
        whitening_jax = jnp.asarray(whitening)

        def trial_sample(point, trial_coefficients):
            # Rebuild the current map inside the differentiable trial map.
            _, _, map_point, _, _ = self._functions()
            before = map_point(point)
            after = apply_increment(before, whitening_jax, trial_coefficients)

            def trial_map(z):
                mapped = map_point(z)
                return apply_increment(mapped, whitening_jax, trial_coefficients)

            jacobian = jax.jacfwd(trial_map)(point)
            determinant = jnp.linalg.det(jacobian)

            def trial_amplitude(z):
                jacobian_z = jax.jacfwd(trial_map)(z)
                _, log_determinant = jnp.linalg.slogdet(jacobian_z)
                a, log_b, displacement, log_c = jnp.asarray(
                    self.baseline_parameters
                )
                b, c = jnp.exp(log_b), jnp.exp(log_c)
                value = (
                    -0.5 * a * jnp.sum(z**2)
                    - 0.25 * b * jnp.sum(z**4)
                    + displacement * jnp.sum(z)
                )
                for first in range(z.size):
                    for second in range(first + 1, z.size):
                        separation = z[first] - z[second]
                        value = value - c * jnp.exp(
                            -0.5 * (separation / self.pair_width) ** 2
                        )
                return value - 0.5 * log_determinant

            gradient_z = jax.grad(trial_amplitude)(point)
            score = jnp.linalg.solve(jacobian.T, gradient_z)
            potential = jnp.sum(
                self.barrier * (after**2 - self.well**2) ** 2 + self.tilt * after
            )
            for first in range(after.size):
                for second in range(first + 1, after.size):
                    separation = after[first] - after[second]
                    potential = potential + self.interaction / jnp.sqrt(
                        separation**2 + self.softening**2
                    )
            energy = potential + self.hbar**2 / (2.0 * self.mass) * jnp.dot(
                score, score
            )
            movement = jnp.dot(after - before, after - before)
            return energy, movement, determinant

        def objective(coefficients):
            energy, movement, determinant = jax.vmap(
                trial_sample, in_axes=(0, None)
            )(base_training, coefficients)
            value = jnp.mean(energy) + self.friction / (2.0 * time_step) * jnp.mean(
                movement
            )
            folding_penalty = 1.0e5 * jnp.mean(jnp.maximum(1.0e-4 - determinant, 0.0) ** 2)
            return value + folding_penalty

        value_and_gradient = jax.jit(jax.value_and_grad(objective))

        def scipy_objective(coefficients):
            value, gradient = value_and_gradient(jnp.asarray(coefficients))
            return float(value), np.asarray(gradient, dtype=float)

        result = minimize(
            scipy_objective,
            np.zeros(whitening.shape[1]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-max_coefficient, max_coefficient)] * whitening.shape[1],
            options={"maxiter": int(maximum_iterations), "ftol": 1.0e-11, "gtol": 1.0e-7},
        )
        accepted = bool(result.fun < current_training["energy"] and np.all(np.isfinite(result.x)))
        if accepted:
            self.transport_whitenings.append(whitening)
            self.transport_coefficients.append(result.x.copy())
        next_state = self.state(self.base_x)
        self.diagnostics = {
            "accepted": accepted,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "objective": float(result.fun),
            "training_energy_before": current_training["energy"],
            "rank": int(whitening.shape[1]),
            "metric_eigenvalues": eigenvalues,
            "coefficient": result.x.copy(),
            "state": next_state,
        }
        return next_state, self.diagnostics

    def run(self, *, time_step=0.08, max_steps=8, warmup=500):
        self.base_x = self.sample_initial(warmup=warmup)
        self.initial_x = self.base_x.copy()
        split = int(self.training_fraction * self.ntraj)
        training = np.arange(split)
        audit = np.arange(split, self.ntraj)
        full_state = self.state(self.base_x)
        training_state = self.state(self.base_x, indices=training)
        audit_state = self.state(self.base_x, indices=audit)
        history = {
            "time": [0.0],
            "energy": [full_state["energy"]],
            "training_energy": [training_state["energy"]],
            "audit_energy": [audit_state["energy"]],
            "audit_standard_error": [audit_state["standard_error"]],
            "minimum_determinant": [full_state["minimum_determinant"]],
            "x": [full_state["x"]],
        }
        for step in range(int(max_steps)):
            full_state, diagnostics = self.proximal_step(time_step=time_step)
            if not diagnostics["accepted"]:
                self.message = "no Jacobian-consistent proximal descent step"
                break
            training_state = self.state(self.base_x, indices=training)
            audit_state = self.state(self.base_x, indices=audit)
            history["time"].append((step + 1) * time_step)
            history["energy"].append(full_state["energy"])
            history["training_energy"].append(training_state["energy"])
            history["audit_energy"].append(audit_state["energy"])
            history["audit_standard_error"].append(audit_state["standard_error"])
            history["minimum_determinant"].append(full_state["minimum_determinant"])
            history["x"].append(full_state["x"])
        else:
            self.message = "maximum Jacobian proximal steps reached"
        self.history = {key: np.asarray(value) for key, value in history.items()}
        self.x = self.history["x"][-1]
        self.energy = float(self.history["audit_energy"][-1])
        self.success = len(self.transport_coefficients) > 0
        return self
