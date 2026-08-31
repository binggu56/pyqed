"""Energy-monotone proximal particle flow with a weak score closure."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .direct_score_flow import DirectOverdampedScoreFlow1D
from .score_corrections import SharedLinearScoreCorrection1D


class ProximalLinearScoreFlow1D(DirectOverdampedScoreFlow1D):
    r"""Jointly update a particle cloud and its fitted linear score.

    A low-dimensional, permutation-invariant map moves the particles.  For
    every trial map the score is refitted by the Stein normal equations and
    the map minimizes

    $$
    E_{\rm weak}[\rho_{k+1}]
    +\frac{\gamma}{2\tau}\langle|R_{k+1}-R_k|^2\rangle.
    $$

    No pointwise quantum force or third amplitude derivative is evaluated.
    """

    def __init__(self, *args, transport_model=None, **kwargs):
        kwargs["closure"] = "linear"
        super().__init__(*args, **kwargs)
        if transport_model is None:
            transport_model = SharedLinearScoreCorrection1D(
                one_body_centers=(-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8),
                one_body_width=1.0,
                pair_widths=(0.8, 1.2, 1.8),
                three_body_scales=(0.12, 0.3),
                regularization=1.0e-6,
            )
        self.transport_model = transport_model
        self.proximal_diagnostics = None

    @staticmethod
    def _clone_linear_model(model):
        return SharedLinearScoreCorrection1D(
            one_body_centers=tuple(model.one_body_centers),
            one_body_width=model.one_body_width,
            pair_widths=tuple(model.pair_widths),
            three_body_scales=tuple(model.three_body_scales),
            regularization=model.regularization,
        )

    def fit_score_state(self, coordinates):
        """Fit the score and return weak and local-energy diagnostics."""
        coordinates = np.asarray(coordinates, dtype=float)
        _, baseline_gradient, baseline_laplacian = self.baseline_terms(coordinates)
        model = self._clone_linear_model(self.linear_model)
        model.fit(coordinates, baseline_gradient, weights=self.weights)
        _, correction_gradient, correction_laplacian = model.terms(coordinates)
        amplitude_gradient = baseline_gradient + correction_gradient
        amplitude_laplacian = baseline_laplacian + correction_laplacian
        kinetic_density = self.hbar**2 / (2.0 * self.mass) * np.sum(
            amplitude_gradient**2, axis=1
        )
        weak_local_energy = self.potential(coordinates) + kinetic_density
        bohm_local_energy = self.potential(coordinates) - self.hbar**2 / (
            2.0 * self.mass
        ) * (
            amplitude_laplacian + np.sum(amplitude_gradient**2, axis=1)
        )
        weak_energy = float(np.dot(self.weights, weak_local_energy))
        local_energy = float(np.dot(self.weights, bohm_local_energy))
        return model, {
            "weak_energy": weak_energy,
            "local_energy": local_energy,
            "energy_gap": local_energy - weak_energy,
            "local_variance": float(
                np.dot(self.weights, (bohm_local_energy - local_energy) ** 2)
            ),
            "weak_variance": float(
                np.dot(self.weights, (weak_local_energy - weak_energy) ** 2)
            ),
            "score_gradient": amplitude_gradient,
            "score_laplacian": amplitude_laplacian,
        }

    def _fit_train_evaluate(self, training, validation):
        """Fit on one cloud and evaluate the energy on another."""
        training = np.asarray(training, dtype=float)
        validation = np.asarray(validation, dtype=float)
        _, training_gradient, _ = self.baseline_terms(training)
        model = self._clone_linear_model(self.linear_model)
        model.fit(
            training,
            training_gradient,
            weights=np.full(training.shape[0], 1.0 / training.shape[0]),
        )
        _, baseline_gradient, baseline_laplacian = self.baseline_terms(validation)
        _, correction_gradient, correction_laplacian = model.terms(validation)
        amplitude_gradient = baseline_gradient + correction_gradient
        amplitude_laplacian = baseline_laplacian + correction_laplacian
        potential = self.potential(validation)
        weak_local = potential + self.hbar**2 / (2.0 * self.mass) * np.sum(
            amplitude_gradient**2, axis=1
        )
        bohm_local = potential - self.hbar**2 / (2.0 * self.mass) * (
            amplitude_laplacian + np.sum(amplitude_gradient**2, axis=1)
        )
        weak_energy = float(np.mean(weak_local))
        local_energy = float(np.mean(bohm_local))
        return model, {
            "weak_energy": weak_energy,
            "local_energy": local_energy,
            "energy_gap": local_energy - weak_energy,
            "local_variance": float(np.var(bohm_local)),
            "weak_variance": float(np.var(weak_local)),
            "local_standard_error": float(
                np.sqrt(np.var(bohm_local) / validation.shape[0])
            ),
        }

    def cross_fitted_state(self, coordinates):
        """Estimate score energy out of sample with two deterministic folds."""
        coordinates = np.asarray(coordinates, dtype=float)
        fold_values = []
        models = []
        for parity in (0, 1):
            training = coordinates[parity::2]
            validation = coordinates[1 - parity :: 2]
            model, state = self._fit_train_evaluate(training, validation)
            fold_values.append(
                (
                    state["weak_energy"],
                    state["local_energy"],
                    state["local_variance"],
                )
            )
            models.append(model)
        values = np.asarray(fold_values)
        return {
            "weak_energy": float(np.mean(values[:, 0])),
            "local_energy": float(np.mean(values[:, 1])),
            "energy_gap": float(np.mean(values[:, 1] - values[:, 0])),
            "local_variance": float(np.mean(values[:, 2])),
            "fold_values": values,
            "models": models,
        }

    def holdout_state(self, coordinates):
        """Fit on even labels and audit only on untouched odd labels."""
        coordinates = np.asarray(coordinates, dtype=float)
        _, state = self._fit_train_evaluate(coordinates[::2], coordinates[1::2])
        return state

    def _whitened_transport_fields(self, coordinates, metric_indices=None):
        _, gradients, _ = self.transport_model.basis_terms(coordinates)
        if metric_indices is None:
            metric_gradients = gradients
        else:
            metric_gradients = gradients[np.asarray(metric_indices)]
        metric = np.einsum(
            "nik,nil->kl", metric_gradients, metric_gradients
        ) / metric_gradients.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        largest = max(float(eigenvalues[-1]), 1.0e-30)
        retained = eigenvalues > 1.0e-7 * largest
        whitening = eigenvectors[:, retained] / np.sqrt(
            eigenvalues[retained]
        )[None, :]
        fields = np.einsum("nik,kl->nil", gradients, whitening)
        return fields, {
            "rank": int(np.count_nonzero(retained)),
            "metric_eigenvalues": eigenvalues,
        }

    def proximal_step(
        self,
        coordinates,
        *,
        time_step=0.04,
        max_rms_displacement=0.08,
        maximum_iterations=35,
        energy_tolerance=1.0e-9,
        consistency_weight=50.0,
        variance_weight=0.05,
        maximum_energy_gap=0.08,
    ):
        """Take one accepted energy-decreasing proximal map step."""
        coordinates = np.asarray(coordinates, dtype=float)
        current_model, current_state = self.fit_score_state(coordinates)
        training_indices = np.arange(coordinates.shape[0])[::2]
        current_training_state = self.cross_fitted_state(
            coordinates[training_indices]
        )
        current_audit_state = self.holdout_state(coordinates)
        fields, field_diagnostics = self._whitened_transport_fields(
            coordinates, metric_indices=training_indices
        )
        rank = fields.shape[-1]
        cache = {}

        def evaluate(coefficients):
            coefficients = np.asarray(coefficients, dtype=float)
            key = coefficients.tobytes()
            if key in cache:
                return cache[key]
            displacement = np.einsum("nir,r->ni", fields, coefficients)
            rms_displacement = float(
                np.sqrt(np.einsum("n,ni,ni->", self.weights, displacement, displacement))
            )
            if rms_displacement > max_rms_displacement:
                current_merit = (
                    current_training_state["weak_energy"]
                    + consistency_weight * current_training_state["energy_gap"] ** 2
                    + variance_weight * current_training_state["local_variance"]
                )
                value = (
                    current_merit
                    + 1.0e3 * (rms_displacement - max_rms_displacement) ** 2
                    + 1.0
                )
                result = (value, None, None, rms_displacement)
                cache[key] = result
                return result
            moved = coordinates + displacement
            if np.max(np.abs(moved)) > 4.5:
                current_merit = (
                    current_training_state["weak_energy"]
                    + consistency_weight * current_training_state["energy_gap"] ** 2
                    + variance_weight * current_training_state["local_variance"]
                )
                result = (current_merit + 1.0e3, None, None, rms_displacement)
                cache[key] = result
                return result
            model, state = self.fit_score_state(moved)
            training_state = self.cross_fitted_state(moved[training_indices])
            audit_state = self.holdout_state(moved)
            movement_cost = self.friction / (2.0 * time_step) * rms_displacement**2
            merit = (
                training_state["weak_energy"]
                + consistency_weight * training_state["energy_gap"] ** 2
                + variance_weight * training_state["local_variance"]
            )
            result = (
                merit + movement_cost,
                moved,
                (model, state, training_state, audit_state),
                rms_displacement,
            )
            cache[key] = result
            return result

        initial = np.zeros(rank)
        result = minimize(
            lambda coefficients: evaluate(coefficients)[0],
            initial,
            method="L-BFGS-B",
            bounds=[(-max_rms_displacement, max_rms_displacement)] * rank,
            options={
                "maxiter": int(maximum_iterations),
                "ftol": 1.0e-11,
                "gtol": 1.0e-7,
                "maxls": 25,
            },
        )
        objective, moved, fitted, rms_displacement = evaluate(result.x)
        current_objective = (
            current_training_state["weak_energy"]
            + consistency_weight * current_training_state["energy_gap"] ** 2
            + variance_weight * current_training_state["local_variance"]
        )
        current_audit_objective = (
            current_audit_state["weak_energy"]
            + consistency_weight * current_audit_state["energy_gap"] ** 2
            + variance_weight * current_audit_state["local_variance"]
        )
        candidate_audit_objective = (
            np.inf
            if fitted is None
            else fitted[3]["weak_energy"]
            + consistency_weight * fitted[3]["energy_gap"] ** 2
            + variance_weight * fitted[3]["local_variance"]
        )
        accepted = (
            moved is not None
            and objective <= current_objective + energy_tolerance
            and candidate_audit_objective
            <= current_audit_objective + energy_tolerance
            and abs(fitted[3]["energy_gap"]) <= maximum_energy_gap
        )
        if accepted:
            next_coordinates = moved
            next_model, next_state, next_training_state, next_audit_state = fitted
        else:
            next_coordinates = coordinates
            next_model, next_state = current_model, current_state
            next_training_state = current_training_state
            next_audit_state = current_audit_state
            rms_displacement = 0.0
            objective = current_objective
        diagnostics = {
            "accepted": bool(accepted),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "objective": float(objective),
            "current_objective": float(current_objective),
            "audit_objective": float(
                candidate_audit_objective if accepted else current_audit_objective
            ),
            "current_audit_objective": float(current_audit_objective),
            "consistency_weight": float(consistency_weight),
            "variance_weight": float(variance_weight),
            "training_state": next_training_state,
            "cross_fitted_state": next_audit_state,
            "audit_state": next_audit_state,
            "rms_displacement": float(rms_displacement),
            "coefficients": result.x.copy(),
            **field_diagnostics,
        }
        self.linear_model = next_model
        return next_coordinates, next_state, diagnostics

    def run(
        self,
        *,
        time_step=0.04,
        max_steps=40,
        max_rms_displacement=0.08,
        energy_tolerance=1.0e-9,
        convergence_tolerance=2.0e-5,
        warmup=500,
    ):
        coordinates = self.sample_initial(warmup=warmup)
        self.initial_x = coordinates.copy()
        _, state = self.fit_score_state(coordinates)
        cross_state = self.holdout_state(coordinates)
        times = [0.0]
        weak_energies = [state["weak_energy"]]
        local_energies = [state["local_energy"]]
        energy_gaps = [state["energy_gap"]]
        local_variances = [state["local_variance"]]
        cross_weak_energies = [cross_state["weak_energy"]]
        cross_local_energies = [cross_state["local_energy"]]
        cross_energy_gaps = [cross_state["energy_gap"]]
        cross_local_variances = [cross_state["local_variance"]]
        displacements = [0.0]
        clouds = [coordinates.copy()]
        accepted_steps = [True]
        time = 0.0
        for _ in range(int(max_steps)):
            coordinates, next_state, diagnostics = self.proximal_step(
                coordinates,
                time_step=time_step,
                max_rms_displacement=max_rms_displacement,
                energy_tolerance=energy_tolerance,
            )
            accepted_steps.append(diagnostics["accepted"])
            displacements.append(diagnostics["rms_displacement"])
            if not diagnostics["accepted"]:
                self.message = "proximal optimizer found no acceptable descent step"
                break
            time += time_step
            times.append(time)
            weak_energies.append(next_state["weak_energy"])
            local_energies.append(next_state["local_energy"])
            energy_gaps.append(next_state["energy_gap"])
            local_variances.append(next_state["local_variance"])
            cross_state = diagnostics["cross_fitted_state"]
            cross_weak_energies.append(cross_state["weak_energy"])
            cross_local_energies.append(cross_state["local_energy"])
            cross_energy_gaps.append(cross_state["energy_gap"])
            cross_local_variances.append(cross_state["local_variance"])
            clouds.append(coordinates.copy())
            if abs(weak_energies[-2] - weak_energies[-1]) < convergence_tolerance:
                self.success = True
                self.message = "proximal weak energy converged"
                break
        else:
            self.message = "maximum proximal steps reached"
        self.x = coordinates
        self.energy = weak_energies[-1]
        self.local_energy = local_energies[-1]
        self.proximal_diagnostics = diagnostics
        self.history = {
            "time": np.asarray(times),
            "weak_energy": np.asarray(weak_energies),
            "local_energy": np.asarray(local_energies),
            "energy_gap": np.asarray(energy_gaps),
            "local_variance": np.asarray(local_variances),
            "cross_weak_energy": np.asarray(cross_weak_energies),
            "cross_local_energy": np.asarray(cross_local_energies),
            "cross_energy_gap": np.asarray(cross_energy_gaps),
            "cross_local_variance": np.asarray(cross_local_variances),
            "rms_displacement": np.asarray(displacements[: len(times)]),
            "accepted": np.asarray(accepted_steps[: len(times)]),
            "x": np.asarray(clouds),
        }
        return self
