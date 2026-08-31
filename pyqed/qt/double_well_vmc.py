"""Difficult-regime VMC benchmark for three particles in a double well."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .direct_score_flow import tilted_double_well_potential
from .score_corrections import global_polynomial_jastrow_terms


def three_body_gaussian_terms(coordinates, *, scale=0.18):
    r"""Return $B_3$, $\nabla B_3$, and $\nabla^2B_3$.

    The invariant feature is

    $$
    B_3=\exp[-\eta\sum_{i<j}(x_i-x_j)^2].
    $$
    """
    coordinates = np.asarray(coordinates, dtype=float)
    particles = coordinates.shape[-1]
    square_sum = np.zeros(coordinates.shape[:-1])
    square_gradient = np.zeros_like(coordinates)
    for first in range(particles):
        for second in range(first + 1, particles):
            separation = coordinates[..., first] - coordinates[..., second]
            square_sum += separation**2
            square_gradient[..., first] += 2.0 * separation
            square_gradient[..., second] -= 2.0 * separation
    feature = np.exp(-float(scale) * square_sum)
    gradient = -float(scale) * feature[..., None] * square_gradient
    laplacian_square_sum = 2.0 * particles * (particles - 1)
    laplacian = feature * (
        float(scale) ** 2 * np.sum(square_gradient**2, axis=-1)
        - float(scale) * laplacian_square_sum
    )
    return feature, gradient, laplacian


def double_well_jastrow_terms(
    coordinates,
    parameters,
    *,
    pair_width=0.8,
    three_body_scale=0.18,
):
    r"""Evaluate a symmetric pair Jastrow with an optional three-body term.

    ``parameters = (a, log_b, displacement, log_c, d3)`` defines the
    amplitude used by :func:`global_polynomial_jastrow_terms` plus $d_3B_3$.
    """
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape[-1] != 5:
        raise ValueError("parameters must contain (a, log_b, displacement, log_c, d3)")
    amplitude, gradient, laplacian = global_polynomial_jastrow_terms(
        coordinates, parameters[:4], pair_width=pair_width
    )
    feature, feature_gradient, feature_laplacian = three_body_gaussian_terms(
        coordinates, scale=three_body_scale
    )
    coefficient = parameters[4]
    return (
        amplitude + coefficient * feature,
        gradient + coefficient * feature_gradient,
        laplacian + coefficient * feature_laplacian,
    )


def three_particle_double_well_potential(
    coordinates,
    *,
    barrier=1.2,
    well=1.6,
    tilt=0.0,
    interaction=1.2,
    softening=0.35,
):
    """Return the symmetric/tilted three-particle benchmark potential."""
    coordinates = np.asarray(coordinates, dtype=float)
    value = np.sum(
        tilted_double_well_potential(
            coordinates, barrier=barrier, well=well, tilt=tilt
        ),
        axis=-1,
    )
    particles = coordinates.shape[-1]
    for first in range(particles):
        for second in range(first + 1, particles):
            separation = coordinates[..., first] - coordinates[..., second]
            value += interaction / np.sqrt(separation**2 + softening**2)
    return value


def double_well_local_energy(
    coordinates,
    parameters,
    *,
    pair_width=0.8,
    three_body_scale=0.18,
    mass=1.0,
    hbar=1.0,
    barrier=1.2,
    well=1.6,
    tilt=0.0,
    interaction=1.2,
    softening=0.35,
):
    """Return $E_L=V+Q$ for the double-well Jastrow state."""
    _, gradient, laplacian = double_well_jastrow_terms(
        coordinates,
        parameters,
        pair_width=pair_width,
        three_body_scale=three_body_scale,
    )
    quantum_potential = -(hbar**2 / (2.0 * mass)) * (
        laplacian + np.sum(gradient**2, axis=-1)
    )
    return three_particle_double_well_potential(
        coordinates,
        barrier=barrier,
        well=well,
        tilt=tilt,
        interaction=interaction,
        softening=softening,
    ) + quantum_potential


def optimize_symmetric_double_well_jastrow(
    *,
    include_three_body=False,
    ngrid=41,
    xmax=4.0,
    pair_width=0.8,
    three_body_scale=0.18,
    barrier=1.2,
    well=1.6,
    tilt=0.0,
    interaction=1.2,
    softening=0.35,
    starts=None,
):
    """Optimize a symmetry-enforced pair or pair-plus-three-body Jastrow."""
    grid = np.linspace(-float(xmax), float(xmax), int(ngrid))
    dx = grid[1] - grid[0]
    x1, x2, x3 = np.meshgrid(grid, grid, grid, indexing="ij")
    coordinates = np.stack((x1, x2, x3), axis=-1)
    potential = three_particle_double_well_potential(
        coordinates,
        barrier=barrier,
        well=well,
        tilt=tilt,
        interaction=interaction,
        softening=softening,
    )

    def unpack(variables):
        variables = np.asarray(variables, dtype=float)
        three_body = variables[3] if include_three_body else 0.0
        return np.array((variables[0], variables[1], 0.0, variables[2], three_body))

    def energy(variables):
        parameters = unpack(variables)
        amplitude, gradient, _ = double_well_jastrow_terms(
            coordinates,
            parameters,
            pair_width=pair_width,
            three_body_scale=three_body_scale,
        )
        density = np.exp(2.0 * (amplitude - np.max(amplitude)))
        density /= np.sum(density) * dx**3
        weak_local = potential + 0.5 * np.sum(gradient**2, axis=-1)
        return float(np.sum(density * weak_local) * dx**3)

    if starts is None:
        starts = (
            (-0.8, np.log(0.35), np.log(0.6), 0.0),
            (-1.5, np.log(0.55), np.log(0.8), 0.0),
            (-2.4, np.log(0.85), np.log(1.0), 0.0),
        )
    variable_size = 4 if include_three_body else 3
    bounds = [(-5.0, 2.0), (-6.0, 2.0), (-6.0, 3.0)]
    if include_three_body:
        bounds.append((-5.0, 5.0))
    results = []
    for start in starts:
        result = minimize(
            energy,
            np.asarray(start[:variable_size], dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1.0e-12, "gtol": 2.0e-8, "maxiter": 160},
        )
        result.full_parameters = unpack(result.x)
        results.append(result)
    best = min(results, key=lambda result: result.fun)
    best.multistart_results = results
    best.grid = grid
    return best


def occupation_probabilities(coordinates):
    """Return probabilities for $n_L=0,1,2,3$ from a sample cloud."""
    coordinates = np.asarray(coordinates, dtype=float)
    occupations = np.sum(coordinates < 0.0, axis=-1)
    counts = np.bincount(occupations.ravel(), minlength=coordinates.shape[-1] + 1)
    return counts / np.sum(counts)


def integrated_autocorrelation_time(series, *, maximum_lag=None):
    """Estimate the initial-positive-sequence integrated autocorrelation time."""
    values = np.asarray(series, dtype=float)
    values = values - np.mean(values)
    variance = float(np.dot(values, values) / values.size)
    if variance < 1.0e-20:
        return np.inf
    if maximum_lag is None:
        maximum_lag = min(values.size // 2, 2000)
    tau = 1.0
    for lag in range(1, int(maximum_lag) + 1):
        correlation = float(
            np.dot(values[:-lag], values[lag:])
            / ((values.size - lag) * variance)
        )
        if correlation <= 0.0:
            break
        tau += 2.0 * correlation
    return tau


class ThreeParticleDoubleWellVMC:
    """Local and reflection-enhanced Markov chains for a fixed Jastrow state."""

    def __init__(
        self,
        parameters,
        *,
        nwalkers=32,
        seed=7,
        pair_width=0.8,
        three_body_scale=0.18,
        barrier=1.2,
        well=1.6,
        tilt=0.0,
        interaction=1.2,
        softening=0.35,
    ):
        self.parameters = np.asarray(parameters, dtype=float)
        self.nwalkers = int(nwalkers)
        self.seed = int(seed)
        self.pair_width = float(pair_width)
        self.three_body_scale = float(three_body_scale)
        self.barrier = float(barrier)
        self.well = float(well)
        self.tilt = float(tilt)
        self.interaction = float(interaction)
        self.softening = float(softening)
        self.history = None
        self.x = None
        self.energy = None
        self.variance = None
        self.acceptance = None
        self.reflection_acceptance = None
        self.occupation_probability = None
        self.energy_autocorrelation = None
        self.sector_autocorrelation = None

    def _amplitude(self, coordinates):
        return double_well_jastrow_terms(
            coordinates,
            self.parameters,
            pair_width=self.pair_width,
            three_body_scale=self.three_body_scale,
        )[0]

    def _local_energy(self, coordinates):
        return double_well_local_energy(
            coordinates,
            self.parameters,
            pair_width=self.pair_width,
            three_body_scale=self.three_body_scale,
            barrier=self.barrier,
            well=self.well,
            tilt=self.tilt,
            interaction=self.interaction,
            softening=self.softening,
        )

    def run(
        self,
        *,
        burn_in=1000,
        sweeps=3000,
        proposal_scale=0.18,
        reflection_probability=0.0,
        initialize_sector=1,
    ):
        rng = np.random.default_rng(self.seed)
        coordinates = np.full((self.nwalkers, 3), self.well)
        coordinates[:, : int(initialize_sector)] *= -1.0
        coordinates += rng.normal(scale=0.08, size=coordinates.shape)
        log_density = 2.0 * self._amplitude(coordinates)
        local_accepted = local_attempted = 0
        reflection_accepted = reflection_attempted = 0
        energy_history = []
        sector_history = []
        clouds = []
        total_sweeps = int(burn_in) + int(sweeps)
        for sweep in range(total_sweeps):
            for particle in range(3):
                proposal = coordinates.copy()
                proposal[:, particle] += rng.normal(
                    scale=float(proposal_scale), size=self.nwalkers
                )
                proposal_log_density = 2.0 * self._amplitude(proposal)
                accept = np.log(rng.random(self.nwalkers)) < (
                    proposal_log_density - log_density
                )
                coordinates[accept] = proposal[accept]
                log_density[accept] = proposal_log_density[accept]
                local_accepted += int(np.count_nonzero(accept))
                local_attempted += self.nwalkers
            reflect = rng.random(self.nwalkers) < float(reflection_probability)
            if np.any(reflect):
                proposal = -coordinates[reflect]
                proposal_log_density = 2.0 * self._amplitude(proposal)
                accept = np.log(rng.random(np.count_nonzero(reflect))) < (
                    proposal_log_density - log_density[reflect]
                )
                reflected_indices = np.flatnonzero(reflect)
                accepted_indices = reflected_indices[accept]
                coordinates[accepted_indices] *= -1.0
                log_density[accepted_indices] = proposal_log_density[accept]
                reflection_accepted += int(np.count_nonzero(accept))
                reflection_attempted += int(np.count_nonzero(reflect))
            if sweep >= int(burn_in):
                local_energy = self._local_energy(coordinates)
                occupation = np.sum(coordinates < 0.0, axis=1)
                energy_history.append(float(np.mean(local_energy)))
                sector_history.append(float(np.mean(occupation - 1.5)))
                clouds.append(coordinates.copy())
        samples = np.asarray(clouds)
        all_coordinates = samples.reshape((-1, 3))
        all_energies = self._local_energy(all_coordinates)
        self.x = coordinates
        self.energy = float(np.mean(all_energies))
        self.variance = float(np.var(all_energies))
        self.acceptance = local_accepted / max(local_attempted, 1)
        self.reflection_acceptance = reflection_accepted / max(
            reflection_attempted, 1
        )
        self.occupation_probability = occupation_probabilities(all_coordinates)
        self.energy_autocorrelation = integrated_autocorrelation_time(energy_history)
        self.sector_autocorrelation = integrated_autocorrelation_time(sector_history)
        self.history = {
            "energy": np.asarray(energy_history),
            "sector": np.asarray(sector_history),
            "x": samples,
        }
        return self
