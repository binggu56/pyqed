"""Direct overdamped quantum-force flow with a particle-fitted score."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import eigsh

from .score_corrections import (
    InvariantNeuralScoreCorrection1D,
    SharedLinearScoreCorrection1D,
    global_polynomial_jastrow_terms,
)


def tilted_double_well_potential(x, *, barrier=0.8, well=1.25, tilt=0.22):
    x = np.asarray(x)
    return barrier * (x**2 - well**2) ** 2 + tilt * x


def tilted_double_well_force(x, *, barrier=0.8, well=1.25, tilt=0.22):
    x = np.asarray(x)
    return -4.0 * barrier * x * (x**2 - well**2) - tilt


def exact_double_well_three_particle_ground_state(
    *,
    barrier=0.8,
    well=1.25,
    tilt=0.22,
    interaction=0.7,
    softening=0.55,
    xmax=3.7,
    ngrid=71,
    mass=1.0,
    hbar=1.0,
):
    """Sparse 3D reference for the direct score-flow benchmark."""
    full_grid = np.linspace(-xmax, xmax, int(ngrid))
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
    potential = np.sum(
        tilted_double_well_potential(
            coordinates, barrier=barrier, well=well, tilt=tilt
        ),
        axis=-1,
    )
    for first, second in ((0, 1), (0, 2), (1, 2)):
        separation = coordinates[..., first] - coordinates[..., second]
        potential += interaction / np.sqrt(separation**2 + softening**2)
    hamiltonian = kinetic + diags(potential.ravel(), format="csr")
    energy, state = eigsh(hamiltonian, k=1, which="SA", tol=1.0e-10)
    psi = state[:, 0].reshape((size, size, size))
    psi /= np.sqrt(np.sum(psi**2) * dx**3)
    if psi[size // 2, size // 2, size // 2] < 0.0:
        psi *= -1.0
    return grid, psi, float(energy[0])


class DirectOverdampedScoreFlow1D:
    r"""Move fixed-weight particles with the full residual quantum force.

    No variational tangent projection, parameter velocity, momentum, or
    Langevin force is used. At every macro-step the score correction is
    refitted to the current cloud and then

    $$
    \gamma\dot R_n=F_{\rm cl}(R_n)+F_Q(R_n)
    $$

    is integrated explicitly.
    """

    def __init__(
        self,
        *,
        ntraj=2048,
        seed=7,
        closure="linear",
        baseline_parameters=(0.3, np.log(0.7), -0.12, np.log(0.3)),
        pair_width=0.8,
        barrier=0.8,
        well=1.25,
        tilt=0.22,
        interaction=0.7,
        softening=0.55,
        mass=1.0,
        hbar=1.0,
        friction=8.0,
        linear_model=None,
        neural_model=None,
    ):
        if closure not in {"baseline", "linear", "neural"}:
            raise ValueError("closure must be 'baseline', 'linear', or 'neural'")
        self.ntraj = int(ntraj)
        self.seed = int(seed)
        self.closure = closure
        self.baseline_parameters = np.asarray(baseline_parameters, dtype=float)
        self.pair_width = float(pair_width)
        self.barrier = float(barrier)
        self.well = float(well)
        self.tilt = float(tilt)
        self.interaction = float(interaction)
        self.softening = float(softening)
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.friction = float(friction)
        self.weights = np.full(self.ntraj, 1.0 / self.ntraj)
        self.linear_model = linear_model or SharedLinearScoreCorrection1D()
        self.neural_model = neural_model or InvariantNeuralScoreCorrection1D(
            hidden_width=12, seed=self.seed, pair_width=self.pair_width
        )
        self.x = None
        self.initial_x = None
        self.energy = None
        self.history = None
        self.success = False
        self.message = "not run"

    def potential(self, coordinates):
        coordinates = np.asarray(coordinates)
        value = np.sum(
            tilted_double_well_potential(
                coordinates, barrier=self.barrier, well=self.well, tilt=self.tilt
            ),
            axis=-1,
        )
        for first, second in ((0, 1), (0, 2), (1, 2)):
            separation = coordinates[..., first] - coordinates[..., second]
            value += self.interaction / np.sqrt(
                separation**2 + self.softening**2
            )
        return value

    def classical_force(self, coordinates):
        coordinates = np.asarray(coordinates)
        force = tilted_double_well_force(
            coordinates, barrier=self.barrier, well=self.well, tilt=self.tilt
        )
        for first, second in ((0, 1), (0, 2), (1, 2)):
            separation = coordinates[..., first] - coordinates[..., second]
            pair_force = self.interaction * separation / (
                separation**2 + self.softening**2
            ) ** 1.5
            force[..., first] += pair_force
            force[..., second] -= pair_force
        return force

    def baseline_terms(self, coordinates):
        return global_polynomial_jastrow_terms(
            coordinates,
            self.baseline_parameters,
            pair_width=self.pair_width,
        )

    def sample_initial(self, *, warmup=500, proposal_scale=0.32):
        rng = np.random.default_rng(self.seed)
        coordinates = rng.normal(scale=1.0, size=(self.ntraj, 3))
        amplitude = self.baseline_terms(coordinates)[0]
        accepted = 0
        for _ in range(int(warmup)):
            proposal = coordinates + rng.normal(
                scale=proposal_scale, size=coordinates.shape
            )
            proposal_amplitude = self.baseline_terms(proposal)[0]
            accept = np.log(rng.random(self.ntraj)) < 2.0 * (
                proposal_amplitude - amplitude
            )
            coordinates[accept] = proposal[accept]
            amplitude[accept] = proposal_amplitude[accept]
            accepted += np.count_nonzero(accept)
        self.acceptance = accepted / (self.ntraj * warmup)
        return coordinates

    def fit_closure(self, coordinates, *, neural_steps=80):
        if self.closure == "linear":
            baseline_gradient = self.baseline_terms(coordinates)[1]
            self.linear_model.fit(
                coordinates, baseline_gradient, weights=self.weights
            )
        elif self.closure == "neural":
            self.neural_model.fit(
                coordinates,
                self.baseline_parameters,
                steps=neural_steps,
                learning_rate=1.0e-4,
                correction_regularization=1.0e-2,
                force_smoothness=1.0e-1,
            )

    def quantum_potential_force(self, coordinates):
        if self.closure == "linear":
            return self.linear_model.quantum_potential_force(
                coordinates,
                self.baseline_parameters,
                pair_width=self.pair_width,
                mass=self.mass,
                hbar=self.hbar,
            )
        if self.closure == "neural":
            return self.neural_model.quantum_potential_force(
                coordinates,
                self.baseline_parameters,
                mass=self.mass,
                hbar=self.hbar,
            )
        _, gradient, laplacian = self.baseline_terms(coordinates)
        prefactor = self.hbar**2 / (2.0 * self.mass)
        quantum_potential = -prefactor * (
            laplacian + np.sum(gradient**2, axis=-1)
        )
        # The baseline path is diagnostic only; use AD through a zero linear correction.
        coefficients = self.linear_model.coefficients
        self.linear_model.coefficients = np.zeros(self.linear_model.size)
        _, force = self.linear_model.quantum_potential_force(
            coordinates,
            self.baseline_parameters,
            pair_width=self.pair_width,
            mass=self.mass,
            hbar=self.hbar,
        )
        self.linear_model.coefficients = coefficients
        return quantum_potential, force

    def state(self, coordinates):
        quantum_potential, quantum_force = self.quantum_potential_force(coordinates)
        local_energy = self.potential(coordinates) + quantum_potential
        residual_force = self.classical_force(coordinates) + quantum_force
        energy = float(np.dot(self.weights, local_energy))
        return {
            "energy": energy,
            "variance": float(np.dot(self.weights, (local_energy - energy) ** 2)),
            "force_rms": float(
                np.sqrt(np.einsum("n,ni,ni->", self.weights, residual_force, residual_force))
            ),
            "max_force": float(np.max(np.linalg.norm(residual_force, axis=1))),
            "local_energy": local_energy,
            "residual_force": residual_force,
        }

    def run(
        self,
        *,
        dt=0.003,
        macro_steps=80,
        tolerance=2.0e-2,
        max_displacement=0.012,
        warmup=500,
        neural_steps=60,
    ):
        coordinates = self.sample_initial(warmup=warmup)
        self.initial_x = coordinates.copy()
        times, energies, variances, force_rms, clouds = [], [], [], [], []
        time = 0.0
        for step_index in range(int(macro_steps) + 1):
            self.fit_closure(coordinates, neural_steps=neural_steps)
            state = self.state(coordinates)
            times.append(time)
            energies.append(state["energy"])
            variances.append(state["variance"])
            force_rms.append(state["force_rms"])
            clouds.append(coordinates.copy())
            if state["force_rms"] < tolerance:
                self.success = True
                self.message = "direct residual force converged"
                break
            if step_index == macro_steps:
                self.message = "maximum macro steps reached"
                break
            velocity = state["residual_force"] / self.friction
            step = min(
                float(dt),
                max_displacement
                / max(float(np.max(np.linalg.norm(velocity, axis=1))), 1.0e-15),
            )
            coordinates = coordinates + step * velocity
            if not np.all(np.isfinite(coordinates)):
                self.message = "direct score flow became non-finite"
                break
            time += step
        self.x = coordinates
        self.energy = energies[-1]
        self.force_rms = force_rms[-1]
        self.history = {
            "time": np.asarray(times),
            "energy": np.asarray(energies),
            "variance": np.asarray(variances),
            "force_rms": np.asarray(force_rms),
            "x": np.asarray(clouds),
        }
        return self


def optimize_global_double_well_jastrow(
    *,
    ngrid=51,
    xmax=3.7,
    pair_width=0.8,
    barrier=0.8,
    well=1.25,
    tilt=0.22,
    interaction=0.7,
    softening=0.55,
):
    """Independently optimize the global baseline by deterministic quadrature."""
    grid = np.linspace(-xmax, xmax, int(ngrid))
    dx = grid[1] - grid[0]
    x1, x2, x3 = np.meshgrid(grid, grid, grid, indexing="ij")
    coordinates = np.stack((x1, x2, x3), axis=-1)
    potential = np.sum(
        tilted_double_well_potential(
            coordinates, barrier=barrier, well=well, tilt=tilt
        ),
        axis=-1,
    )
    for first, second in ((0, 1), (0, 2), (1, 2)):
        separation = coordinates[..., first] - coordinates[..., second]
        potential += interaction / np.sqrt(separation**2 + softening**2)

    def energy(parameters):
        amplitude, gradient, _ = global_polynomial_jastrow_terms(
            coordinates, parameters, pair_width=pair_width
        )
        density = np.exp(2.0 * (amplitude - np.max(amplitude)))
        density /= np.sum(density) * dx**3
        kinetic = 0.5 * np.sum(gradient**2, axis=-1)
        return float(np.sum(density * (potential + kinetic)) * dx**3)

    result = minimize(
        energy,
        np.asarray((0.3, np.log(0.7), -0.12, np.log(0.3))),
        method="BFGS",
        options={"gtol": 1.0e-8, "maxiter": 150},
    )
    return result
