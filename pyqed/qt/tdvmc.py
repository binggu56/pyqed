"""Real-time variational Monte Carlo with continuity-consistent trajectories."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def anharmonic_double_well(x, barrier=0.18, well=1.6, tilt=0.0):
    r"""Return $V(x)=\lambda(x^2-a^2)^2+\eta x$."""
    x = np.asarray(x)
    return barrier * (x**2 - well**2) ** 2 + tilt * x


class ComplexJastrowTDVMC1D:
    r"""Holomorphic complex-Jastrow TDVMC in one spatial dimension.

    The logarithm of the wavefunction is

    $$
    J_\theta(x)=-\kappa x^4+\sum_a\theta_a f_a(x),
    $$

    where the real features contain linear and quadratic polynomials followed
    by Gaussian radial functions.  The fixed negative quartic envelope keeps
    every member of the complex parameter family normalizable.
    """

    def __init__(
        self,
        *,
        xmin=-6.0,
        xmax=6.0,
        ngrid=1601,
        centers=(-2.8, -2.1, -1.4, -0.7, 0.0, 0.7, 1.4, 2.1, 2.8),
        feature_width=0.72,
        envelope=2.0e-4,
        mass=1.0,
        hbar=1.0,
        barrier=0.18,
        well=1.6,
        tilt=0.0,
        metric_cutoff=1.0e-10,
        metric_shift=1.0e-4,
    ):
        if ngrid < 101 or ngrid % 2 == 0:
            raise ValueError("ngrid must be an odd integer of at least 101")
        if xmax <= xmin or feature_width <= 0.0 or envelope <= 0.0:
            raise ValueError("invalid grid, feature width, or envelope")
        if mass <= 0.0 or hbar <= 0.0:
            raise ValueError("mass and hbar must be positive")
        self.grid = np.linspace(xmin, xmax, int(ngrid))
        self.dx = float(self.grid[1] - self.grid[0])
        self.centers = np.asarray(centers, dtype=float)
        self.feature_width = float(feature_width)
        self.envelope = float(envelope)
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.barrier = float(barrier)
        self.well = float(well)
        self.tilt = float(tilt)
        self.metric_cutoff = float(metric_cutoff)
        self.metric_shift = float(metric_shift)
        self.potential_grid = anharmonic_double_well(
            self.grid, self.barrier, self.well, self.tilt
        )

        self.theta = None
        self.trajectories = None
        self.history = None
        self.success = False
        self.message = "not run"

    @property
    def nparams(self):
        return 2 + self.centers.size

    def features(self, x):
        """Return Jastrow features and their first two derivatives."""
        x = np.asarray(x, dtype=float)
        displacement = x[..., None] - self.centers
        width2 = self.feature_width**2
        radial = np.exp(-0.5 * displacement**2 / width2)
        radial_first = -displacement * radial / width2
        radial_second = (displacement**2 / width2**2 - 1.0 / width2) * radial
        values = np.concatenate((x[..., None], x[..., None] ** 2, radial), axis=-1)
        first = np.concatenate(
            (np.ones_like(x)[..., None], 2.0 * x[..., None], radial_first), axis=-1
        )
        second = np.concatenate(
            (np.zeros_like(x)[..., None], 2.0 * np.ones_like(x)[..., None], radial_second),
            axis=-1,
        )
        return values, first, second

    def log_derivatives(self, x, theta):
        r"""Return $J$, $\partial_xJ$, and $\partial_x^2J$."""
        x = np.asarray(x, dtype=float)
        theta = np.asarray(theta, dtype=complex)
        values, first, second = self.features(x)
        return (
            -self.envelope * x**4 + values @ theta,
            -4.0 * self.envelope * x**3 + first @ theta,
            -12.0 * self.envelope * x**2 + second @ theta,
        )

    def initial_parameters(self, *, center=-1.55, width=0.48, momentum=0.75):
        """Return a localized Gaussian-like initial complex Jastrow state."""
        theta = np.zeros(self.nparams, dtype=complex)
        theta[0] = center / (2.0 * width**2) + 1j * momentum / self.hbar
        theta[1] = -1.0 / (4.0 * width**2)
        return theta

    def density(self, theta, x=None):
        """Evaluate the normalized variational density on the main grid."""
        if x is not None and not np.array_equal(np.asarray(x), self.grid):
            raise ValueError("normalized density is currently tabulated on the main grid")
        log_psi, _, _ = self.log_derivatives(self.grid, theta)
        shifted = np.real(log_psi) - np.max(np.real(log_psi))
        rho = np.exp(2.0 * shifted)
        rho /= np.trapezoid(rho, self.grid)
        return rho

    def wavefunction(self, theta):
        """Return the normalized complex variational wavefunction."""
        log_psi, _, _ = self.log_derivatives(self.grid, theta)
        shifted = log_psi - np.max(np.real(log_psi))
        psi = np.exp(shifted)
        psi /= np.sqrt(np.trapezoid(np.abs(psi) ** 2, self.grid))
        return psi

    def local_energy(self, x, theta):
        """Evaluate the complex local energy."""
        _, first, second = self.log_derivatives(x, theta)
        kinetic = -(self.hbar**2 / (2.0 * self.mass)) * (second + first**2)
        return kinetic + anharmonic_double_well(
            x, self.barrier, self.well, self.tilt
        )

    def _moments(self, theta, points=None):
        if points is None:
            points = self.grid
            weights = self.density(theta) * self.dx
            weights[[0, -1]] *= 0.5
            weights /= np.sum(weights)
        else:
            points = np.asarray(points, dtype=float)
            weights = np.full(points.size, 1.0 / points.size)
        values, _, _ = self.features(points)
        local_energy = self.local_energy(points, theta)
        mean_values = weights @ values
        centered = values - mean_values
        metric = np.einsum("n,na,nb->ab", weights, centered, centered)
        force = np.einsum("n,na,n->a", weights, centered, local_energy)
        energy = np.dot(weights, local_energy)
        return metric, force, energy

    def tdvp_velocity(self, theta, points=None):
        """Return the real-time TDVP parameter velocity and diagnostics."""
        metric, force, energy = self._moments(theta, points)
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        scale = max(float(eigenvalues[-1]), 1.0e-14)
        threshold = self.metric_cutoff * scale
        retained = eigenvalues > threshold
        if not np.any(retained):
            raise np.linalg.LinAlgError("the TDVP metric has zero numerical rank")
        projected = eigenvectors[:, retained].T @ force
        inverse_force = eigenvectors[:, retained] @ (
            projected / (eigenvalues[retained] + self.metric_shift * scale)
        )
        velocity = -1j * inverse_force / self.hbar
        diagnostics = {
            "energy": float(np.real(energy)),
            "energy_imaginary": float(np.imag(energy)),
            "metric_rank": int(np.count_nonzero(retained)),
            "metric_condition": float(
                (eigenvalues[-1] + self.metric_shift * scale)
                / (eigenvalues[retained][0] + self.metric_shift * scale)
            ),
        }
        return velocity, diagnostics

    def bohmian_velocity(self, x, theta):
        """Return the phase-gradient Bohmian velocity."""
        _, first, _ = self.log_derivatives(x, theta)
        return self.hbar * np.imag(first) / self.mass

    def continuity_state(self, theta, theta_dot):
        r"""Return the exact 1D tangent lift and continuity diagnostics.

        The returned velocity is the minimum-kinetic-energy field satisfying
        $\partial_t\rho_\theta+\partial_x(\rho_\theta v_T)=0$ on the grid.
        """
        x = self.grid
        rho = self.density(theta)
        values, _, _ = self.features(x)
        means = np.trapezoid(rho[:, None] * values, x, axis=0)
        density_rate_ratio = 2.0 * np.real((values - means) @ theta_dot)
        rho_dot = rho * density_rate_ratio
        integrated = cumulative_trapezoid(rho_dot, x, initial=0.0)
        cdf = cumulative_trapezoid(rho, x, initial=0.0)
        cdf /= cdf[-1]
        integrated -= cdf * integrated[-1]
        floor = np.max(rho) * 1.0e-12
        tangent_velocity = -integrated / np.maximum(rho, floor)

        _, log_first, log_second = self.log_derivatives(x, theta)
        bohmian_velocity = self.hbar * np.imag(log_first) / self.mass
        bohmian_divergence = self.hbar * np.imag(log_second) / self.mass
        residual_ratio = (
            density_rate_ratio
            + 2.0 * np.real(log_first) * bohmian_velocity
            + bohmian_divergence
        )
        residual_rms = float(np.sqrt(np.trapezoid(rho * residual_ratio**2, x)))
        correction_rms = float(
            np.sqrt(np.trapezoid(rho * (tangent_velocity - bohmian_velocity) ** 2, x))
        )
        return {
            "rho": rho,
            "rho_dot": rho_dot,
            "cdf": cdf,
            "bohmian_velocity": bohmian_velocity,
            "tangent_velocity": tangent_velocity,
            "velocity_correction": tangent_velocity - bohmian_velocity,
            "continuity_residual_rms": residual_rms,
            "velocity_correction_rms": correction_rms,
        }

    def quantile_trajectories(self, theta, quantiles):
        r"""Map fixed labels to the quantiles of $|\psi_\theta|^2$."""
        rho = self.density(theta)
        cdf = cumulative_trapezoid(rho, self.grid, initial=0.0)
        cdf /= cdf[-1]
        return np.interp(np.asarray(quantiles), cdf, self.grid)

    def energy(self, theta, points=None):
        """Return the real variational energy."""
        return self.tdvp_velocity(theta, points)[1]["energy"]


def split_operator_step(psi, potential, dx, dt, *, mass=1.0, hbar=1.0):
    """Apply one second-order periodic split-operator step."""
    psi = np.asarray(psi, dtype=complex)
    phase_v = np.exp(-0.5j * np.asarray(potential) * dt / hbar)
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(psi.size, d=dx)
    phase_t = np.exp(-0.5j * hbar * wave_numbers**2 * dt / mass)
    return phase_v * np.fft.ifft(phase_t * np.fft.fft(phase_v * psi))
