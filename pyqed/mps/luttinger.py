"""Exact continuum solution of a nonlocal Tomonaga-Luttinger model.

The interaction is represented by the same real-space exponential terms used
by the continuum cLETTA examples,

    V(x) = sum_j g_j exp(-kappa_j |x|).

After bosonization the model is Gaussian, so each momentum mode is solved by a
Bogoliubov rotation.  This module provides the resulting momentum-dependent
Luttinger parameter, mode velocity, dispersion, and static structure factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import roots_legendre

from .cmps import (
    ContinuousMPS,
    canonical_parameter_size,
    pack_canonical_parameters,
)


@dataclass
class ExponentialLuttingerModel:
    r"""Nonlocal spinless Luttinger liquid with exponential interactions.

    The bosonized Hamiltonian for each positive momentum is

    $$
    H_k = |k|\left[A(k)(b_{R,k}^\dagger b_{R,k}
        +b_{L,k}^\dagger b_{L,k})
        +B(k)(b_{R,k}^\dagger b_{L,k}^\dagger+b_{L,k}b_{R,k})\right],
    $$

    where $A=v_F+\widetilde V/(2\pi)$ and
    $B=\widetilde V/(2\pi)$.  The interaction strengths therefore have units
    such that $\widetilde V(k)$ is a velocity when $\hbar=1$.
    """

    decay_rates: np.ndarray
    strengths: np.ndarray
    fermi_velocity: float = 1.0

    def __post_init__(self):
        self.decay_rates = np.atleast_1d(np.asarray(self.decay_rates, dtype=float))
        self.strengths = np.atleast_1d(np.asarray(self.strengths, dtype=float))
        self.fermi_velocity = float(self.fermi_velocity)
        if self.decay_rates.ndim != 1 or self.strengths.ndim != 1:
            raise ValueError("decay_rates and strengths must be one-dimensional.")
        if self.decay_rates.shape != self.strengths.shape:
            raise ValueError("decay_rates and strengths must have the same shape.")
        if self.decay_rates.size == 0:
            raise ValueError("at least one exponential interaction term is required.")
        if np.any(self.decay_rates <= 0.0):
            raise ValueError("all decay rates must be positive.")
        if self.fermi_velocity <= 0.0:
            raise ValueError("fermi_velocity must be positive.")

    def interaction_real_space(self, distance):
        """Return $V(x)$ for scalar or array distances."""
        distance = np.asarray(distance, dtype=float)
        values = np.exp(
            -np.abs(distance)[..., np.newaxis] * self.decay_rates
        ) @ self.strengths
        return _scalar_if_scalar(values, distance)

    def interaction_momentum(self, momentum):
        r"""Return the exact Fourier transform $\widetilde V(k)$."""
        momentum = np.asarray(momentum, dtype=float)
        values = (
            2.0
            * self.decay_rates
            * self.strengths
            / (momentum[..., np.newaxis] ** 2 + self.decay_rates**2)
        ).sum(axis=-1)
        return _scalar_if_scalar(values, momentum)

    def _stable_denominator(self, momentum):
        interaction = np.asarray(self.interaction_momentum(momentum), dtype=float)
        denominator = self.fermi_velocity + interaction / np.pi
        if np.any(denominator <= 0.0):
            raise ValueError(
                "the interaction makes the Luttinger Hamiltonian unstable: "
                "v_F + V(k) / pi must remain positive."
            )
        return denominator

    def luttinger_parameter(self, momentum):
        r"""Return $K(k)=[v_F/(v_F+\widetilde V(k)/\pi)]^{1/2}$."""
        momentum_array = np.asarray(momentum, dtype=float)
        values = np.sqrt(
            self.fermi_velocity / self._stable_denominator(momentum_array)
        )
        return _scalar_if_scalar(values, momentum_array)

    def mode_velocity(self, momentum):
        r"""Return $u(k)=[v_F(v_F+\widetilde V(k)/\pi)]^{1/2}$."""
        momentum_array = np.asarray(momentum, dtype=float)
        values = np.sqrt(
            self.fermi_velocity * self._stable_denominator(momentum_array)
        )
        return _scalar_if_scalar(values, momentum_array)

    def dispersion(self, momentum):
        r"""Return the exact collective-mode dispersion $\omega(k)=u(k)|k|$."""
        momentum_array = np.asarray(momentum, dtype=float)
        values = np.abs(momentum_array) * self.mode_velocity(momentum_array)
        return _scalar_if_scalar(values, momentum_array)

    def static_structure_factor(self, momentum):
        r"""Return $\langle\delta n_k\delta n_{-k}\rangle/L$.

        With $\delta n=-\partial_x\phi/\pi$, the convention used here is

        $$
        S(k)=\frac{K(k)|k|}{2\pi}.
        $$
        """
        momentum_array = np.asarray(momentum, dtype=float)
        values = (
            np.abs(momentum_array)
            * self.luttinger_parameter(momentum_array)
            / (2.0 * np.pi)
        )
        return _scalar_if_scalar(values, momentum_array)

    def exact_squeezing(self, momentum):
        r"""Return the exact two-mode Bogoliubov squeezing $\theta_*(k)$.

        The convention is $K(k)=\exp[-2\theta_*(k)]$, so repulsive
        interactions have positive squeezing.
        """
        momentum_array = np.asarray(momentum, dtype=float)
        interaction = np.asarray(
            self.interaction_momentum(momentum_array), dtype=float
        )
        argument = 1.0 + interaction / (np.pi * self.fermi_velocity)
        if np.any(argument <= 0.0):
            raise ValueError(
                "the interaction makes the Luttinger Hamiltonian unstable: "
                "1 + V(k) / (pi v_F) must remain positive."
            )
        values = 0.25 * np.log(argument)
        return _scalar_if_scalar(values, momentum_array)

    def ground_state_energy_shift_density(
        self,
        *,
        momentum_max=np.inf,
        epsabs=1.0e-12,
        epsrel=1.0e-11,
    ):
        r"""Return the exact Bogoliubov vacuum-energy shift per unit length.

        The energy is normal ordered relative to the noninteracting Fermi sea:

        $$
        \frac{\Delta E_0}{L}=\int_0^\infty\frac{dk}{2\pi}
        k\,[u(k)-A(k)].
        $$

        The algebraically equivalent $-B(k)^2/[u(k)+A(k)]$ form is evaluated
        to avoid cancellation at large momentum.
        """
        upper = float(momentum_max)
        if upper <= 0.0:
            raise ValueError("momentum_max must be positive.")

        def integrand(momentum):
            interaction = float(self.interaction_momentum(momentum))
            b_value = interaction / (2.0 * np.pi)
            a_value = self.fermi_velocity + b_value
            velocity = float(self.mode_velocity(momentum))
            return -momentum * b_value * b_value / (
                2.0 * np.pi * (velocity + a_value)
            )

        value, error = quad(
            integrand,
            0.0,
            upper,
            epsabs=float(epsabs),
            epsrel=float(epsrel),
            limit=300,
        )
        return float(value), float(error)

    def density_correlation(
        self,
        distances,
        *,
        uv_cutoff,
        points=12000,
        integration_max=None,
    ):
        r"""Return the UV-regulated connected density correlation.

        An exponential momentum regulator is used:

        $$
        C_{nn}(x)=\frac{1}{2\pi^2}\int_0^\infty dk\,
        kK(k)e^{-k/\Lambda}\cos(kx).
        $$
        """
        cutoff = float(uv_cutoff)
        if cutoff <= 0.0:
            raise ValueError("uv_cutoff must be positive.")
        if integration_max is None:
            integration_max = 18.0 * cutoff
        integration_max = float(integration_max)
        if integration_max <= 0.0:
            raise ValueError("integration_max must be positive.")
        points = int(points)
        if points < 2:
            raise ValueError("points must be at least two.")

        distances_array = np.asarray(distances, dtype=float)
        momentum = np.linspace(0.0, integration_max, points)
        weight = (
            momentum
            * self.luttinger_parameter(momentum)
            * np.exp(-momentum / cutoff)
            / (2.0 * np.pi**2)
        )
        flat_distances = distances_array.reshape(-1)
        values = np.trapezoid(
            weight[:, np.newaxis]
            * np.cos(momentum[:, np.newaxis] * flat_distances[np.newaxis, :]),
            momentum,
            axis=0,
        ).reshape(distances_array.shape)
        return _scalar_if_scalar(values, distances_array)


@dataclass
class GaussianLuttingerCLETTA:
    r"""Finite-memory Gaussian cLETTA for the nonlocal Luttinger liquid.

    Each auxiliary channel contributes a real-space exponential memory.  In
    momentum space the variational squeezing is

    $$
    \theta_M(k)=\sum_{\mu=1}^M s_\mu
    \frac{\lambda_\mu^2}{k^2+\lambda_\mu^2}.
    $$

    This is the Gaussian sector of cLETTA: all contractions are analytic, so
    no Fock cutoff or HEOM hierarchy truncation is needed.
    """

    model: ExponentialLuttingerModel
    amplitudes: np.ndarray
    decay_rates: np.ndarray
    energy_shift_density: float | None = None
    exact_energy_shift_density: float | None = None
    success: bool | None = None
    message: str | None = None
    nfev: int = 0
    nit: int = 0

    def __post_init__(self):
        self.amplitudes = np.atleast_1d(np.asarray(self.amplitudes, dtype=float))
        self.decay_rates = np.atleast_1d(np.asarray(self.decay_rates, dtype=float))
        if self.amplitudes.shape != self.decay_rates.shape:
            raise ValueError("amplitudes and decay_rates must have the same shape.")
        if np.any(self.decay_rates <= 0.0):
            raise ValueError("all cLETTA decay rates must be positive.")

    @property
    def num_modes(self):
        return int(self.amplitudes.size)

    def squeezing(self, momentum):
        r"""Return the finite-memory squeezing kernel $\theta_M(k)$."""
        momentum_array = np.asarray(momentum, dtype=float)
        if self.num_modes == 0:
            values = np.zeros_like(momentum_array)
        else:
            basis = self.decay_rates**2 / (
                momentum_array[..., np.newaxis] ** 2 + self.decay_rates**2
            )
            values = basis @ self.amplitudes
        return _scalar_if_scalar(values, momentum_array)

    def luttinger_parameter(self, momentum):
        r"""Return the variational $K_M(k)=\exp[-2\theta_M(k)]$."""
        momentum_array = np.asarray(momentum, dtype=float)
        values = np.exp(-2.0 * self.squeezing(momentum_array))
        return _scalar_if_scalar(values, momentum_array)

    def static_structure_factor(self, momentum):
        """Return the variational static structure factor."""
        momentum_array = np.asarray(momentum, dtype=float)
        values = (
            np.abs(momentum_array)
            * self.luttinger_parameter(momentum_array)
            / (2.0 * np.pi)
        )
        return _scalar_if_scalar(values, momentum_array)

    def evaluate_energy(self, *, epsabs=1.0e-11, epsrel=1.0e-10):
        """Adaptively evaluate and store the variational energy density."""
        value, _error = quad(
            lambda momentum: _gaussian_luttinger_energy_integrand(
                self.model,
                momentum,
                float(self.squeezing(momentum)),
            ),
            0.0,
            np.inf,
            epsabs=float(epsabs),
            epsrel=float(epsrel),
            limit=300,
        )
        self.energy_shift_density = float(value)
        if self.exact_energy_shift_density is None:
            self.exact_energy_shift_density = (
                self.model.ground_state_energy_shift_density()[0]
            )
        return self.energy_shift_density

    @classmethod
    def optimize(
        cls,
        model,
        *,
        num_modes,
        seed_states=(),
        restarts=4,
        seed=0,
        maxiter=500,
        quadrature_points=700,
        momentum_scale=None,
    ):
        """Variationally optimize a finite-memory Gaussian cLETTA state."""
        num_modes = int(num_modes)
        if num_modes < 0:
            raise ValueError("num_modes must be non-negative.")
        exact_energy = model.ground_state_energy_shift_density()[0]
        if num_modes == 0:
            state = cls(
                model=model,
                amplitudes=np.zeros(0),
                decay_rates=np.zeros(0),
                energy_shift_density=0.0,
                exact_energy_shift_density=exact_energy,
                success=True,
                message="free-boson vacuum",
            )
            return state

        momentum, weights = _infinite_momentum_quadrature(
            quadrature_points,
            model.decay_rates if momentum_scale is None else momentum_scale,
        )
        interaction = np.asarray(model.interaction_momentum(momentum), dtype=float)
        b_value = interaction / (2.0 * np.pi)
        a_value = model.fermi_velocity + b_value
        prefactor = weights * momentum / (2.0 * np.pi)
        scale = _momentum_scale(
            model.decay_rates if momentum_scale is None else momentum_scale
        )
        rate_min = scale * 1.0e-3
        rate_max = scale * 1.0e3
        bounds = [(-6.0, 6.0)] * num_modes + [
            (np.log(rate_min), np.log(rate_max))
        ] * num_modes

        def value_gradient(parameters):
            amplitudes = parameters[:num_modes]
            rates = np.exp(parameters[num_modes:])
            denominator = momentum[:, np.newaxis] ** 2 + rates**2
            basis = rates**2 / denominator
            theta = basis @ amplitudes
            sinh_two = np.sinh(2.0 * theta)
            cosh_two = np.cosh(2.0 * theta)
            energy_density = (
                2.0 * a_value * np.sinh(theta) ** 2 - b_value * sinh_two
            )
            value = float(np.dot(prefactor, energy_density))
            derivative_theta = prefactor * (
                2.0 * a_value * sinh_two - 2.0 * b_value * cosh_two
            )
            gradient_amplitudes = basis.T @ derivative_theta
            derivative_log_rates = (
                2.0
                * rates**2
                * momentum[:, np.newaxis] ** 2
                / denominator**2
            )
            gradient_rates = (
                derivative_log_rates * amplitudes[np.newaxis, :]
            ).T @ derivative_theta
            gradient = np.concatenate([gradient_amplitudes, gradient_rates])
            return value, gradient

        initial_parameters = []
        for state in seed_states:
            if not isinstance(state, cls) or state.model is not model:
                continue
            if state.num_modes > num_modes:
                continue
            missing = num_modes - state.num_modes
            if missing:
                new_rates = np.geomspace(scale * 0.7, scale * 1.7, missing)
                rates = np.concatenate([state.decay_rates, new_rates])
                amplitudes = np.concatenate([state.amplitudes, np.zeros(missing)])
            else:
                rates = state.decay_rates
                amplitudes = state.amplitudes
            initial_parameters.append(
                np.concatenate([amplitudes, np.log(rates)])
            )

        fit_momentum = np.concatenate(
            [[0.0], np.geomspace(scale * 1.0e-4, scale * 1.0e2, 500)]
        )
        fit_rates = np.geomspace(scale * 0.45, scale * 2.2, num_modes)
        fit_basis = fit_rates**2 / (
            fit_momentum[:, np.newaxis] ** 2 + fit_rates**2
        )
        fit_amplitudes = np.linalg.lstsq(
            fit_basis,
            model.exact_squeezing(fit_momentum),
            rcond=None,
        )[0]
        initial_parameters.append(
            np.concatenate([fit_amplitudes, np.log(fit_rates)])
        )

        rng = np.random.default_rng(seed)
        target_restarts = max(int(restarts), len(initial_parameters))
        while len(initial_parameters) < target_restarts:
            amplitudes = fit_amplitudes + rng.normal(scale=0.08, size=num_modes)
            log_rates = np.log(fit_rates) + rng.normal(scale=0.7, size=num_modes)
            initial_parameters.append(np.concatenate([amplitudes, log_rates]))

        best = None
        total_nfev = 0
        total_nit = 0
        for parameters in initial_parameters:
            result = minimize(
                value_gradient,
                np.asarray(parameters, dtype=float),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": int(maxiter), "ftol": 1.0e-14, "gtol": 1.0e-10},
            )
            total_nfev += int(result.nfev)
            total_nit += int(result.nit)
            if best is None or result.fun < best.fun:
                best = result

        amplitudes = np.asarray(best.x[:num_modes], dtype=float)
        rates = np.exp(np.asarray(best.x[num_modes:], dtype=float))
        order = np.argsort(rates)
        state = cls(
            model=model,
            amplitudes=amplitudes[order],
            decay_rates=rates[order],
            energy_shift_density=float(best.fun),
            exact_energy_shift_density=float(exact_energy),
            success=bool(best.success),
            message=str(best.message),
            nfev=total_nfev,
            nit=total_nit,
        )
        state.evaluate_energy()
        return state


def cmps_luttinger_spectra(state, momentum):
    r"""Return the even normal and anomalous spectra of a one-field cMPS.

    The local continuum field combines the two chiral sectors as
    $a_{k>0}=b_{R,k}$ and $a_{k<0}=b_{L,-k}$.  Disconnected zero-momentum
    contributions are projected out because the Luttinger Hamiltonian carries
    an overall factor $|k|$.
    """
    if not isinstance(state, ContinuousMPS) or state.num_fields != 1:
        raise ValueError("Luttinger spectra require a single-field ContinuousMPS.")
    momentum_array = np.asarray(momentum, dtype=float)
    if np.any(momentum_array < 0.0):
        raise ValueError("momentum must be non-negative.")

    transfer = np.asarray(state.transfer_matrix(), dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eig(transfer)
    inverse_vectors = np.linalg.inv(eigenvectors)
    dominant = int(np.argmax(np.real(eigenvalues)))
    leading_value = eigenvalues[dominant]
    right = eigenvectors[:, dominant]
    left_row = inverse_vectors[dominant, :]

    dim = state.bond_dim
    eye = np.eye(dim, dtype=np.complex128)
    field = np.asarray(state.r, dtype=np.complex128)
    ket_insertion = np.kron(field, eye)
    bra_insertion = np.kron(eye, field.conj())
    initial = ket_insertion @ right

    mode_initial = inverse_vectors @ initial
    normal_coefficients = (left_row @ bra_insertion @ eigenvectors) * mode_initial
    anomalous_coefficients = (left_row @ ket_insertion @ eigenvectors) * mode_initial
    normal_coefficients[dominant] = 0.0
    anomalous_coefficients[dominant] = 0.0
    poles = eigenvalues - leading_value
    poles[dominant] = np.inf

    flat_momentum = momentum_array.reshape(-1)
    minus_denominator = poles[:, np.newaxis] - 1.0j * flat_momentum[np.newaxis, :]
    plus_denominator = poles[:, np.newaxis] + 1.0j * flat_momentum[np.newaxis, :]
    normal = np.real(
        -np.sum(normal_coefficients[:, np.newaxis] / minus_denominator, axis=0)
        -np.sum(normal_coefficients[:, np.newaxis] / plus_denominator, axis=0)
    )
    anomalous = np.real(
        -np.sum(anomalous_coefficients[:, np.newaxis] / minus_denominator, axis=0)
        -np.sum(anomalous_coefficients[:, np.newaxis] / plus_denominator, axis=0)
    )
    normal = normal.reshape(momentum_array.shape)
    anomalous = anomalous.reshape(momentum_array.shape)
    if momentum_array.ndim == 0:
        return float(normal), float(anomalous)
    return normal, anomalous


def cmps_luttinger_energy_shift_density(
    model,
    state,
    *,
    quadrature_points=180,
    momentum_scale=None,
):
    r"""Return $\langle H\rangle/L$ for a matrix cMPS or cLETTA state."""
    momentum, weights = _infinite_momentum_quadrature(
        quadrature_points,
        model.decay_rates if momentum_scale is None else momentum_scale,
    )
    normal, anomalous = cmps_luttinger_spectra(state, momentum)
    interaction = np.asarray(model.interaction_momentum(momentum), dtype=float)
    b_value = interaction / (2.0 * np.pi)
    a_value = model.fermi_velocity + b_value
    integrand = momentum * (
        2.0 * a_value * normal + 2.0 * b_value * anomalous
    ) / (2.0 * np.pi)
    return float(np.dot(weights, integrand))


def cmps_luttinger_parameter(state, momentum):
    r"""Return the cMPS quadrature covariance $K(k)=1+2n(k)+2m(k)$."""
    normal, anomalous = cmps_luttinger_spectra(state, momentum)
    return 1.0 + 2.0 * (normal + anomalous)


def optimize_luttinger_cletta(
    model,
    *,
    bond_dim,
    num_modes,
    depth=1,
    memory_decay_rates=None,
    optimize_memory_rates=True,
    seed_states=(),
    seed_parameters=(),
    restarts=5,
    seed=0,
    maxiter=350,
    quadrature_points=160,
    regularization=1.0e-10,
    initial_scale=0.25,
    tie_scale=0.08,
    rate_bounds=None,
):
    r"""Optimize a genuine $(D,M)$ matrix cLETTA for the Luttinger model.

    The optimized variables are the left-canonical cMPS core $(Q,R)$, the
    matrix-valued memory ties $G_\mu$, and optionally their decay rates.  The
    returned object is the explicit finite-depth cLETTA ``ContinuousMPS``.
    """
    bond_dim = int(bond_dim)
    num_modes = int(num_modes)
    depth = int(depth)
    if bond_dim < 1:
        raise ValueError("bond_dim must be positive.")
    if num_modes < 0:
        raise ValueError("num_modes must be non-negative.")
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    if num_modes == 0:
        optimize_memory_rates = False

    if memory_decay_rates is None:
        if num_modes:
            scale = _momentum_scale(model.decay_rates)
            reference_rates = np.geomspace(scale * 0.7, scale * 1.7, num_modes)
        else:
            reference_rates = np.zeros(0, dtype=float)
    else:
        reference_rates = np.atleast_1d(
            np.asarray(memory_decay_rates, dtype=float)
        )
        if reference_rates.shape != (num_modes,):
            raise ValueError("memory_decay_rates must have length num_modes.")
        if np.any(reference_rates <= 0.0):
            raise ValueError("memory_decay_rates must be positive.")

    scale = _momentum_scale(model.decay_rates)
    if rate_bounds is None:
        lower_rate, upper_rate = scale * 1.0e-3, scale * 1.0e3
    else:
        lower_rate, upper_rate = map(float, rate_bounds)
        if not (0.0 < lower_rate < upper_rate):
            raise ValueError("rate_bounds must satisfy 0 < lower < upper.")

    base_size = canonical_parameter_size(bond_dim)
    tie_size = num_modes * bond_dim * bond_dim
    parameter_size = base_size + tie_size + (
        num_modes if optimize_memory_rates else 0
    )

    def pack(base_theta, ties, rates):
        pieces = [
            np.asarray(base_theta, dtype=float).reshape(-1),
            np.asarray(ties, dtype=float).reshape(-1),
        ]
        if optimize_memory_rates:
            pieces.append(np.log(np.clip(rates, lower_rate, upper_rate)))
        return np.concatenate(pieces)

    def unpack(parameters):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.size != parameter_size:
            raise ValueError(
                f"parameter size {parameters.size} does not match {parameter_size}."
            )
        base_theta = parameters[:base_size]
        offset = base_size
        ties = parameters[offset : offset + tie_size].reshape(
            num_modes, bond_dim, bond_dim
        )
        offset += tie_size
        if optimize_memory_rates:
            rates = np.exp(
                np.clip(
                    parameters[offset : offset + num_modes],
                    np.log(lower_rate),
                    np.log(upper_rate),
                )
            )
        else:
            rates = reference_rates
        return base_theta, ties, rates

    def build_state(parameters):
        base_theta, ties, rates = unpack(parameters)
        base = ContinuousMPS.from_canonical_parameters(base_theta, bond_dim)
        if num_modes:
            state = base.cletta_memory_state(ties, rates, depth=depth)
        else:
            state = base
            state.cletta_base = base
            state.cletta_tie_matrices = np.zeros(
                (0, bond_dim, bond_dim), dtype=float
            )
            state.cletta_decay_rates = np.zeros(0, dtype=float)
            state.cletta_frequencies = np.zeros(0, dtype=float)
            state.cletta_depth = depth
        state.cletta_parameters = np.asarray(parameters, dtype=float).copy()
        return state

    rng = np.random.default_rng(seed)
    candidates = [np.asarray(values, dtype=float) for values in seed_parameters]
    zero_ties = np.zeros((num_modes, bond_dim, bond_dim), dtype=float)
    for state in seed_states:
        if not isinstance(state, ContinuousMPS):
            continue
        base = state.cletta_base if state.cletta_base is not None else state
        if base.bond_dim > bond_dim or base.theta is None:
            continue
        if base.bond_dim == bond_dim:
            base_theta = base.theta
        else:
            old_dim = base.bond_dim
            old_field = np.asarray(base.r, dtype=float)
            old_drift = np.asarray(base.canonical_drift(), dtype=float)
            drift = np.zeros((bond_dim, bond_dim), dtype=float)
            field = np.zeros((bond_dim, bond_dim), dtype=float)
            drift[:old_dim, :old_dim] = old_drift
            field[:old_dim, :old_dim] = old_field
            embed_rng = np.random.default_rng(104729 * bond_dim + old_dim)
            noise = 1.0e-4 * embed_rng.normal(size=(bond_dim, bond_dim))
            drift += noise - noise.T
            field += 1.0e-4 * embed_rng.normal(size=(bond_dim, bond_dim))
            base_theta = pack_canonical_parameters(drift, field)
        old_ties = state.cletta_tie_matrices
        old_rates = state.cletta_decay_rates
        old_modes = 0 if old_ties is None else int(len(old_ties))
        if old_modes > num_modes:
            continue
        ties = np.zeros_like(zero_ties)
        rates = np.array(reference_rates, copy=True)
        if old_modes:
            old_dim = base.bond_dim
            ties[:old_modes, :old_dim, :old_dim] = np.asarray(
                old_ties,
                dtype=float,
            )
            rates[:old_modes] = np.asarray(old_rates, dtype=float)
        candidates.append(pack(base_theta, ties, rates))

    target_restarts = max(int(restarts) + len(candidates), len(candidates))
    while len(candidates) < target_restarts:
        base_theta = ContinuousMPS.random_canonical_parameters(
            bond_dim,
            rng=rng,
            scale=float(initial_scale),
        )
        ties = float(tie_scale) * rng.normal(
            size=(num_modes, bond_dim, bond_dim)
        )
        rates = reference_rates * np.exp(
            0.45 * rng.normal(size=num_modes)
        )
        candidates.append(pack(base_theta, ties, rates))

    evaluations = 0

    def physical_energy(parameters):
        state = build_state(parameters)
        return cmps_luttinger_energy_shift_density(
            model,
            state,
            quadrature_points=quadrature_points,
        )

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        parameters = np.asarray(parameters, dtype=float)
        if not np.all(np.isfinite(parameters)):
            return 1.0e30
        try:
            energy = physical_energy(parameters)
        except (
            FloatingPointError,
            np.linalg.LinAlgError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            return 1.0e30
        if not np.isfinite(energy):
            return 1.0e30
        return float(energy) + float(regularization) * float(
            np.dot(parameters, parameters)
        )

    matrix_bounds = [(-4.0, 4.0)] * (base_size + tie_size)
    bounds = list(matrix_bounds)
    if optimize_memory_rates:
        bounds.extend(
            [(np.log(lower_rate), np.log(upper_rate))] * num_modes
        )

    results = []
    for initial in candidates:
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "maxls": 60, "ftol": 1.0e-12},
        )
        results.append(result)

    final_candidates = [
        (np.asarray(result.x, dtype=float), result.success, str(result.message))
        for result in results
    ]
    final_candidates.extend(
        (initial, True, "seeded matrix cLETTA candidate")
        for initial in candidates
    )
    best_parameters = None
    best_energy = np.inf
    best_success = False
    best_message = "no valid matrix cLETTA candidate"
    for parameters, success, message in final_candidates:
        try:
            energy = physical_energy(parameters)
        except (
            FloatingPointError,
            np.linalg.LinAlgError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue
        if np.isfinite(energy) and energy < best_energy:
            best_parameters = parameters
            best_energy = float(energy)
            best_success = bool(success)
            best_message = message
    if best_parameters is None:
        raise FloatingPointError("no valid matrix cLETTA candidate found.")

    state = build_state(best_parameters)
    state.energy = best_energy
    state.luttinger_energy_shift_density = best_energy
    state.luttinger_exact_energy_shift_density = (
        model.ground_state_energy_shift_density()[0]
    )
    state.luttinger_bond_dim = bond_dim
    state.luttinger_num_modes = num_modes
    state.luttinger_depth = depth
    state.success = best_success
    state.message = best_message
    state.nfev = evaluations
    state.algorithm = "matrix-cletta-luttinger-L-BFGS-B"
    return state


def _scalar_if_scalar(values, argument):
    values = np.asarray(values)
    if np.asarray(argument).ndim == 0:
        return values.item()
    return values


def _momentum_scale(scale_or_rates):
    values = np.atleast_1d(np.asarray(scale_or_rates, dtype=float))
    if values.size == 0 or np.any(values <= 0.0):
        raise ValueError("momentum scale must be positive.")
    return float(np.sqrt(np.min(values) * np.max(values)))


def _infinite_momentum_quadrature(points, scale_or_rates):
    points = int(points)
    if points < 16:
        raise ValueError("quadrature_points must be at least 16.")
    scale = _momentum_scale(scale_or_rates)
    nodes, weights = _unit_legendre_quadrature(points)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    momentum = scale * unit_nodes / (1.0 - unit_nodes)
    jacobian = scale / (1.0 - unit_nodes) ** 2
    return momentum, unit_weights * jacobian


@lru_cache(maxsize=16)
def _unit_legendre_quadrature(points):
    nodes, weights = roots_legendre(int(points))
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _gaussian_luttinger_energy(model, momentum, weights, theta):
    interaction = np.asarray(model.interaction_momentum(momentum), dtype=float)
    b_value = interaction / (2.0 * np.pi)
    a_value = model.fermi_velocity + b_value
    energy_density = 2.0 * a_value * np.sinh(theta) ** 2 - b_value * np.sinh(
        2.0 * theta
    )
    return float(np.dot(weights * momentum / (2.0 * np.pi), energy_density))


def _gaussian_luttinger_energy_integrand(model, momentum, theta):
    interaction = float(model.interaction_momentum(momentum))
    b_value = interaction / (2.0 * np.pi)
    a_value = model.fermi_velocity + b_value
    energy = 2.0 * a_value * np.sinh(theta) ** 2 - b_value * np.sinh(
        2.0 * theta
    )
    return momentum * energy / (2.0 * np.pi)


__all__ = [
    "ExponentialLuttingerModel",
    "GaussianLuttingerCLETTA",
    "cmps_luttinger_energy_shift_density",
    "cmps_luttinger_parameter",
    "cmps_luttinger_spectra",
    "optimize_luttinger_cletta",
]
