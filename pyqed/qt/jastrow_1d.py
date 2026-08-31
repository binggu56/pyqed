"""One-dimensional trajectory models with an analytic Jastrow quantum force.

The projected solver transports fixed quantiles of a positive Jastrow density.
Consequently, its particle velocities are restricted to the tangent space of
the density manifold.  The legacy solver is a cleaned-up implementation of the
polynomial quantum-force closure in ``pyqed/qt/1D/AHO.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.special import ndtri


def quartic_potential(x, anharmonicity=0.4):
    r"""Return $V(x)=x^2/2+\epsilon x^4/4$."""
    x = np.asarray(x)
    return 0.5 * x**2 + 0.25 * anharmonicity * x**4


def quartic_force(x, anharmonicity=0.4):
    """Return the classical force $-V'(x)$."""
    x = np.asarray(x)
    return -x - anharmonicity * x**3


def exact_quartic_ground_state(
    *, anharmonicity=0.4, mass=1.0, hbar=1.0, xmax=8.0, ngrid=2001
):
    """Solve the quartic oscillator on a uniform finite-difference grid."""
    if ngrid < 5 or ngrid % 2 == 0:
        raise ValueError("ngrid must be an odd integer of at least five")
    x = np.linspace(-xmax, xmax, ngrid)
    dx = x[1] - x[0]
    interior = x[1:-1]
    interior_size = ngrid - 2
    kinetic = -(hbar**2 / (2.0 * mass * dx**2)) * diags(
        (
            np.ones(interior_size - 1),
            -2.0 * np.ones(interior_size),
            np.ones(interior_size - 1),
        ),
        (-1, 0, 1),
        format="csr",
    )
    hamiltonian = kinetic + diags(
        quartic_potential(interior, anharmonicity), format="csr"
    )
    energy, state = eigsh(hamiltonian, k=1, which="SA")
    psi = np.zeros_like(x)
    psi[1:-1] = state[:, 0]
    psi /= np.sqrt(np.trapezoid(psi**2, x))
    if psi[ngrid // 2] < 0.0:
        psi *= -1.0
    return x, psi, float(energy[0])


class ProjectedJastrow1D:
    r"""Overdamped trajectory relaxation on a two-parameter Jastrow manifold.

    The positive wavefunction is

    $$
    \psi(x)=Z^{-1/2}\exp[-a x^2/2-b h_s(x)],
    \qquad
    h_s(x)=\frac{(x^2+s^2)^{3/2}-s^3}{3},
    $$

    with $a=\exp(\theta_0)$ and $b=\exp(\theta_1)$.  The $h_s$ term has the
    correct $|x|^3$ asymptotic form for a quartic oscillator.
    """

    def __init__(
        self,
        *,
        ntraj=256,
        mass=1.0,
        hbar=1.0,
        anharmonicity=0.4,
        smoothing=0.35,
        friction=1.0,
        xmax=8.0,
        ngrid=4001,
        regularization=1.0e-10,
    ):
        if ntraj < 4:
            raise ValueError("ntraj must be at least four")
        if mass <= 0.0 or hbar <= 0.0 or smoothing <= 0.0 or friction <= 0.0:
            raise ValueError("mass, hbar, smoothing, and friction must be positive")
        self.ntraj = int(ntraj)
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.anharmonicity = float(anharmonicity)
        self.smoothing = float(smoothing)
        self.friction = float(friction)
        self.regularization = float(regularization)
        self.grid = np.linspace(-xmax, xmax, int(ngrid))
        self.quantiles = (np.arange(self.ntraj) + 0.5) / self.ntraj

        self.theta = None
        self.x = None
        self.energy = None
        self.gradient = None
        self.history = None
        self.success = False
        self.message = "not run"

    def _shape(self, x):
        s = self.smoothing
        radius = np.sqrt(x**2 + s**2)
        h = (radius**3 - s**3) / 3.0
        dh = x * radius
        ddh = (2.0 * x**2 + s**2) / radius
        dddh = x * (2.0 * x**2 + 3.0 * s**2) / radius**3
        return h, dh, ddh, dddh

    def log_amplitude_derivatives(self, x, theta=None):
        r"""Return $A$, $A'$, $A''$, and $A'''$ for $A=\log|\psi|$."""
        if theta is None:
            theta = self.theta
        theta = np.asarray(theta, dtype=float)
        a, b = np.exp(theta)
        h, dh, ddh, dddh = self._shape(np.asarray(x))
        amplitude = -0.5 * a * np.asarray(x) ** 2 - b * h
        return (
            amplitude,
            -a * np.asarray(x) - b * dh,
            -a - b * ddh,
            -b * dddh,
        )

    def quantum_potential_force(self, x, theta=None):
        """Evaluate the analytic Bohm potential and force."""
        _, da, dda, ddda = self.log_amplitude_derivatives(x, theta)
        prefactor = self.hbar**2 / (2.0 * self.mass)
        quantum_potential = -prefactor * (dda + da**2)
        quantum_force = prefactor * (ddda + 2.0 * da * dda)
        return quantum_potential, quantum_force

    def _density_state(self, theta):
        x = self.grid
        amplitude, da, dda, _ = self.log_amplitude_derivatives(x, theta)
        shifted = amplitude - np.max(amplitude)
        rho = np.exp(2.0 * shifted)
        rho /= np.trapezoid(rho, x)

        cdf = cumulative_trapezoid(rho, x, initial=0.0)
        cdf /= cdf[-1]
        trajectories = np.interp(self.quantiles, cdf, x)

        a, b = np.exp(theta)
        h, _, _, _ = self._shape(x)
        observables = np.array((-0.5 * a * x**2, -b * h))
        means = np.array([np.trapezoid(rho * obs, x) for obs in observables])

        tangents = []
        for obs, mean in zip(observables, means):
            drho = 2.0 * rho * (obs - mean)
            integrated = cumulative_trapezoid(drho, x, initial=0.0)
            integrated -= cdf * integrated[-1]
            floor = np.max(rho) * 1.0e-14
            field = -integrated / np.maximum(rho, floor)
            tangents.append(np.interp(trajectories, x, field))
        tangents = np.asarray(tangents).T

        _, quantum_force = self.quantum_potential_force(trajectories, theta)
        residual_force = quartic_force(
            trajectories, self.anharmonicity
        ) + quantum_force
        metric = self.mass * tangents.T @ tangents / self.ntraj
        generalized_force = tangents.T @ residual_force / self.ntraj

        potential_energy = np.trapezoid(
            rho * quartic_potential(x, self.anharmonicity), x
        )
        kinetic_energy = self.hbar**2 / (2.0 * self.mass) * np.trapezoid(
            rho * da**2, x
        )
        local_energy = quartic_potential(x, self.anharmonicity) - (
            self.hbar**2 / (2.0 * self.mass) * (dda + da**2)
        )
        energy = float(potential_energy + kinetic_energy)
        variance = float(np.trapezoid(rho * (local_energy - energy) ** 2, x))
        return {
            "rho": rho,
            "x": trajectories,
            "tangents": tangents,
            "metric": metric,
            "force": generalized_force,
            "energy": energy,
            "variance": variance,
            "residual_force": residual_force,
        }

    def energy_at(self, theta):
        """Return the normalized variational energy at ``theta``."""
        return self._density_state(np.asarray(theta, dtype=float))["energy"]

    def energy_gradient(self, theta=None, step=2.0e-5):
        """Return a centered finite-difference energy gradient."""
        if theta is None:
            theta = self.theta
        theta = np.asarray(theta, dtype=float)
        gradient = np.empty_like(theta)
        for a in range(theta.size):
            displacement = np.zeros_like(theta)
            displacement[a] = step
            gradient[a] = (
                self.energy_at(theta + displacement)
                - self.energy_at(theta - displacement)
            ) / (2.0 * step)
        return gradient

    def _velocity(self, theta):
        state = self._density_state(theta)
        metric = state["metric"]
        scale = max(float(np.trace(metric)) / len(theta), 1.0)
        regularized = metric + self.regularization * scale * np.eye(len(theta))
        velocity = np.linalg.solve(
            self.friction * regularized, state["force"]
        )
        return velocity, state

    def run(
        self,
        *,
        theta0=(np.log(1.5), np.log(0.08)),
        dt=0.05,
        max_steps=1000,
        tolerance=1.0e-8,
        record_every=5,
    ):
        """Relax the parameters with a backtracked projected Euler flow."""
        theta = np.asarray(theta0, dtype=float).copy()
        times, energies, variances, parameters, forces, trajectories = [], [], [], [], [], []
        time = 0.0

        for step_index in range(max_steps + 1):
            velocity, state = self._velocity(theta)
            if step_index % record_every == 0 or step_index == max_steps:
                times.append(time)
                energies.append(state["energy"])
                variances.append(state["variance"])
                parameters.append(theta.copy())
                forces.append(state["force"].copy())
                trajectories.append(state["x"].copy())

            if np.linalg.norm(state["force"], ord=np.inf) < tolerance:
                self.success = True
                self.message = "projected force converged"
                break
            if step_index == max_steps:
                self.message = "maximum steps reached"
                break

            trial_dt = float(dt)
            while trial_dt > dt * 1.0e-8:
                trial_theta = theta + trial_dt * velocity
                if np.any(np.abs(trial_theta) > 20.0):
                    trial_dt *= 0.5
                    continue
                trial_energy = self.energy_at(trial_theta)
                if (
                    np.isfinite(trial_energy)
                    and trial_energy <= state["energy"] + 1.0e-13
                ):
                    theta = trial_theta
                    time += trial_dt
                    break
                trial_dt *= 0.5
            else:
                self.message = "line search failed"
                break

        final_state = self._density_state(theta)
        self.theta = theta
        self.x = final_state["x"]
        self.energy = final_state["energy"]
        self.gradient = self.energy_gradient(theta)
        self.history = {
            "time": np.asarray(times),
            "energy": np.asarray(energies),
            "variance": np.asarray(variances),
            "theta": np.asarray(parameters),
            "projected_force": np.asarray(forces),
            "x": np.asarray(trajectories),
        }
        return self

    def density(self, x=None):
        """Return the normalized final variational density."""
        if self.theta is None:
            raise RuntimeError("run the solver before requesting its density")
        if x is None or np.array_equal(np.asarray(x), self.grid):
            return self._density_state(self.theta)["rho"]
        amplitude, *_ = self.log_amplitude_derivatives(np.asarray(x), self.theta)
        amplitude_grid, *_ = self.log_amplitude_derivatives(self.grid, self.theta)
        normalization = np.trapezoid(
            np.exp(2.0 * (amplitude_grid - np.max(amplitude_grid))), self.grid
        ) * np.exp(2.0 * np.max(amplitude_grid))
        return np.exp(2.0 * amplitude) / normalization


class LegacyPolynomialQTM1D:
    """Frictional Bohmian trajectories with a polynomial score closure."""

    def __init__(
        self,
        *,
        ntraj=1024,
        mass=1.0,
        anharmonicity=0.4,
        friction=2.0,
        degree=3,
    ):
        self.ntraj = int(ntraj)
        self.mass = float(mass)
        self.anharmonicity = float(anharmonicity)
        self.friction = float(friction)
        self.degree = int(degree)
        self.x = None
        self.p = None
        self.r = None
        self.energy = None
        self.history = None
        self.success = False
        self.message = "not run"

    def _quantum_terms(self, x, p, r):
        cr = np.polynomial.polynomial.polyfit(x, r, self.degree)
        cp = np.polynomial.polynomial.polyfit(x, p, self.degree)
        dr = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(cr)
        )
        ddr = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(cr, 2)
        )
        dp = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(cp)
        )
        ddp = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(cp, 2)
        )
        quantum_force = (2.0 * r * dr + ddr) / (2.0 * self.mass)
        score_force = -(2.0 * r * dp + ddp) / (2.0 * self.mass)
        quantum_energy = -np.mean(r**2 + dr) / (2.0 * self.mass)
        return quantum_energy, quantum_force, score_force

    def run(
        self,
        *,
        width=2.0,
        center=0.0,
        dt=0.002,
        steps=7500,
        record_every=25,
    ):
        quantiles = (np.arange(self.ntraj) + 0.5) / self.ntraj
        x = center + ndtri(quantiles) / np.sqrt(2.0 * width)
        p = np.zeros_like(x)
        r = -width * (x - center)
        half_dt = 0.5 * dt
        times, energies, trajectories = [], [], []

        potential = quartic_potential(x, self.anharmonicity)
        quantum_energy, quantum_force, score_force = self._quantum_terms(x, p, r)
        for step_index in range(steps + 1):
            if step_index % record_every == 0 or step_index == steps:
                kinetic = np.mean(p**2) / (2.0 * self.mass)
                energies.append(kinetic + np.mean(potential) + quantum_energy)
                times.append(step_index * dt)
                trajectories.append(x.copy())
            if step_index == steps:
                break

            p += (
                quartic_force(x, self.anharmonicity)
                + quantum_force
                - self.friction * p
            ) * half_dt
            r += score_force * half_dt
            x += p * dt / self.mass
            potential = quartic_potential(x, self.anharmonicity)
            quantum_energy, quantum_force, score_force = self._quantum_terms(x, p, r)
            p += (
                quartic_force(x, self.anharmonicity)
                + quantum_force
                - self.friction * p
            ) * half_dt
            r += score_force * half_dt

            if not np.all(np.isfinite(x)):
                self.message = "trajectory propagation became non-finite"
                break

        self.x, self.p, self.r = x, p, r
        self.energy = float(
            np.mean(p**2) / (2.0 * self.mass)
            + np.mean(quartic_potential(x, self.anharmonicity))
            + quantum_energy
        )
        self.history = {
            "time": np.asarray(times),
            "energy": np.asarray(energies),
            "x": np.asarray(trajectories),
        }
        self.success = np.all(np.isfinite(x))
        if self.success:
            self.message = "propagation completed"
        return self
