"""Invariant local bases for Monte Carlo trajectory transport."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _wendland(distance, centers, widths):
    """Return compact $C^2$ radial functions and radial derivatives."""
    displacement = distance[..., None] - centers
    q = np.abs(displacement) / widths
    inside = q < 1.0
    one_minus_q = np.maximum(1.0 - q, 0.0)
    values = one_minus_q**4 * (4.0 * q + 1.0)
    derivatives = (
        -20.0
        * q
        * one_minus_q**3
        * np.sign(displacement)
        / widths
    )
    return np.where(inside, values, 0.0), np.where(inside, derivatives, 0.0)


@dataclass(frozen=True)
class SharedRadialTransportBasis:
    r"""Shared local invariant scalar features and their gradients.

    One-body features are sums of compact radial functions of $|r_i|$;
    two-body features are sums over $|r_i-r_j|$; and optional three-body
    features are symmetric products over the three edges of each local
    triangle.  Feature counts are independent of particle number.
    """

    one_body_centers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    pair_centers: tuple[float, ...] = (0.0, 0.6, 1.2, 1.8, 2.4, 3.0)
    one_body_width: float = 1.0
    pair_width: float = 1.1
    three_body_centers: tuple[float, ...] = ()
    three_body_width: float = 1.4
    distance_epsilon: float = 1.0e-8

    def __post_init__(self):
        if self.one_body_width <= 0.0 or self.pair_width <= 0.0:
            raise ValueError("radial feature widths must be positive")
        if self.three_body_centers and self.three_body_width <= 0.0:
            raise ValueError("three-body feature width must be positive")

    @property
    def size(self):
        return (
            len(self.one_body_centers)
            + len(self.pair_centers)
            + len(self.three_body_centers)
        )

    @property
    def labels(self):
        labels = [f"one-body RBF {center:g}" for center in self.one_body_centers]
        labels += [f"pair RBF {center:g}" for center in self.pair_centers]
        labels += [
            f"three-body RBF {center:g}" for center in self.three_body_centers
        ]
        return labels

    def values_and_gradients(self, coordinates):
        """Evaluate features and Cartesian gradients.

        Parameters
        ----------
        coordinates
            Array with shape ``(samples, particles, spatial_dimension)``.
        """
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 3:
            raise ValueError("coordinates must have shape (samples, particles, dim)")
        samples, particles, spatial_dimension = coordinates.shape
        values, gradients = [], []
        epsilon = self.distance_epsilon

        if self.one_body_centers:
            radii = np.sqrt(np.sum(coordinates**2, axis=2) + epsilon**2)
            radial_values, radial_derivatives = _wendland(
                radii,
                np.asarray(self.one_body_centers),
                np.full(len(self.one_body_centers), self.one_body_width),
            )
            one_values = np.sum(radial_values, axis=1)
            directions = coordinates / radii[:, :, None]
            one_gradients = np.einsum(
                "npc,npd->npdc", radial_derivatives, directions
            )
            values.append(one_values)
            gradients.append(one_gradients)

        pair_indices = list(combinations(range(particles), 2))
        pair_distances = None
        pair_directions = None
        if pair_indices:
            first = np.asarray([pair[0] for pair in pair_indices])
            second = np.asarray([pair[1] for pair in pair_indices])
            displacements = coordinates[:, first] - coordinates[:, second]
            pair_distances = np.sqrt(
                np.sum(displacements**2, axis=2) + epsilon**2
            )
            pair_directions = displacements / pair_distances[:, :, None]

        if self.pair_centers:
            pair_values = np.zeros((samples, len(self.pair_centers)))
            pair_gradients = np.zeros(
                (samples, particles, spatial_dimension, len(self.pair_centers))
            )
            if pair_indices:
                radial_values, radial_derivatives = _wendland(
                    pair_distances,
                    np.asarray(self.pair_centers),
                    np.full(len(self.pair_centers), self.pair_width),
                )
                pair_values = np.sum(radial_values, axis=1)
                edge_gradients = np.einsum(
                    "npc,npd->npdc", radial_derivatives, pair_directions
                )
                for edge, (particle_i, particle_j) in enumerate(pair_indices):
                    pair_gradients[:, particle_i] += edge_gradients[:, edge]
                    pair_gradients[:, particle_j] -= edge_gradients[:, edge]
            values.append(pair_values)
            gradients.append(pair_gradients)

        if self.three_body_centers:
            three_values = np.zeros((samples, len(self.three_body_centers)))
            three_gradients = np.zeros(
                (samples, particles, spatial_dimension, len(self.three_body_centers))
            )
            edge_lookup = {pair: edge for edge, pair in enumerate(pair_indices)}
            centers = np.asarray(self.three_body_centers)
            widths = np.full(len(centers), self.three_body_width)
            for particle_i, particle_j, particle_k in combinations(
                range(particles), 3
            ):
                edges = (
                    edge_lookup[(particle_i, particle_j)],
                    edge_lookup[(particle_i, particle_k)],
                    edge_lookup[(particle_j, particle_k)],
                )
                edge_values, edge_derivatives = [], []
                for edge in edges:
                    radial_values, radial_derivatives = _wendland(
                        pair_distances[:, edge], centers, widths
                    )
                    edge_values.append(radial_values)
                    edge_derivatives.append(radial_derivatives)
                product = edge_values[0] * edge_values[1] * edge_values[2]
                three_values += product
                for local_edge, (left, right) in enumerate(
                    (
                        (particle_i, particle_j),
                        (particle_i, particle_k),
                        (particle_j, particle_k),
                    )
                ):
                    others = [index for index in range(3) if index != local_edge]
                    radial_derivative = (
                        edge_derivatives[local_edge]
                        * edge_values[others[0]]
                        * edge_values[others[1]]
                    )
                    direction = pair_directions[:, edges[local_edge]]
                    contribution = np.einsum(
                        "nc,nd->ndc", radial_derivative, direction
                    )
                    three_gradients[:, left] += contribution
                    three_gradients[:, right] -= contribution
            values.append(three_values)
            gradients.append(three_gradients)

        if not values:
            return (
                np.empty((samples, 0)),
                np.empty((samples, particles, spatial_dimension, 0)),
                [],
            )
        return (
            np.concatenate(values, axis=1),
            np.concatenate(gradients, axis=3),
            self.labels,
        )


def weak_poisson_objective(values, gradients, scores, weights=None, mass=1.0):
    r"""Return the minimized Galerkin objective and coefficients."""
    values = np.asarray(values, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    scores = np.asarray(scores, dtype=float)
    samples = values.shape[0]
    if weights is None:
        weights = np.full(samples, 1.0 / samples)
    weights = np.asarray(weights, dtype=float)
    centered = values - np.einsum("n,nk->k", weights, values)[None, :]
    kinetic = np.einsum(
        "n,npdk,npdl->kl", weights, gradients / mass, gradients
    )
    target = np.einsum("n,nk,na->ka", weights, centered, scores)
    scale = max(float(np.trace(kinetic)) / max(kinetic.shape[0], 1), 1.0)
    system = kinetic + 1.0e-10 * scale * np.eye(kinetic.shape[0])
    coefficients = np.linalg.solve(system, target)
    objective = 0.5 * np.sum(coefficients * (kinetic @ coefficients)) - np.sum(
        coefficients * target
    )
    residual = kinetic @ coefficients - target
    return float(objective), coefficients, residual


def select_three_body_features(
    coordinates,
    scores,
    base_basis,
    candidate_centers,
    *,
    max_features=2,
    minimum_relative_improvement=1.0e-3,
    weights=None,
    mass=1.0,
):
    """Greedily add three-body features when they improve the weak objective."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape[1] < 3:
        return base_basis, []
    selected, remaining = [], list(candidate_centers)
    current = base_basis
    values, gradients, _ = current.values_and_gradients(coordinates)
    objective, _, _ = weak_poisson_objective(
        values, gradients, scores, weights, mass
    )
    baseline_scale = max(abs(objective), 1.0e-14)
    selection_history = []
    while remaining and len(selected) < int(max_features):
        trials = []
        for center in remaining:
            trial = SharedRadialTransportBasis(
                one_body_centers=current.one_body_centers,
                pair_centers=current.pair_centers,
                one_body_width=current.one_body_width,
                pair_width=current.pair_width,
                three_body_centers=tuple(selected + [center]),
                three_body_width=current.three_body_width,
                distance_epsilon=current.distance_epsilon,
            )
            trial_values, trial_gradients, _ = trial.values_and_gradients(
                coordinates
            )
            trial_objective, _, _ = weak_poisson_objective(
                trial_values, trial_gradients, scores, weights, mass
            )
            trials.append((objective - trial_objective, center, trial, trial_objective))
        improvement, center, trial, trial_objective = max(trials, key=lambda item: item[0])
        relative_improvement = improvement / baseline_scale
        if relative_improvement < minimum_relative_improvement:
            break
        selected.append(center)
        remaining.remove(center)
        current, objective = trial, trial_objective
        selection_history.append(
            {
                "center": float(center),
                "improvement": float(improvement),
                "relative_improvement": float(relative_improvement),
            }
        )
    return current, selection_history

