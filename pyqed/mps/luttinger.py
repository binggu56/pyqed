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
from scipy.special import k0, roots_legendre

from .cmps import (
    ContinuousMPS,
    _dominant_sparse_biorthogonal_pair,
    canonical_parameter_size,
    pack_canonical_parameters,
    skew_pairs,
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
        return _luttinger_density_correlation(
            self.luttinger_parameter,
            distances,
            uv_cutoff=uv_cutoff,
            points=points,
            integration_max=integration_max,
        )


@dataclass
class CoulombLuttingerModel:
    r"""Spinless Luttinger liquid with an unscreened softened Coulomb kernel.

    The real-space interaction and its exact Fourier transform are

    $$
    V(x)=\frac{g}{\sqrt{x^2+a^2}},\qquad
    \widetilde V(k)=2gK_0(a|k|).
    $$

    The softening length ``a`` regulates only the short-distance singularity;
    the $1/|x|$ tail and logarithmic infrared singularity remain intact.
    """

    coupling: float
    softening: float = 1.0
    fermi_velocity: float = 1.0

    def __post_init__(self):
        self.coupling = float(self.coupling)
        self.softening = float(self.softening)
        self.fermi_velocity = float(self.fermi_velocity)
        if self.coupling <= 0.0:
            raise ValueError("coupling must be positive.")
        if self.softening <= 0.0:
            raise ValueError("softening must be positive.")
        if self.fermi_velocity <= 0.0:
            raise ValueError("fermi_velocity must be positive.")

    @property
    def decay_rates(self):
        """Return the inverse softening length as a quadrature scale hint."""
        return np.array([1.0 / self.softening])

    def interaction_real_space(self, distance):
        """Return the softened Coulomb interaction $V(x)$."""
        distance_array = np.asarray(distance, dtype=float)
        values = self.coupling / np.sqrt(
            distance_array**2 + self.softening**2
        )
        return _scalar_if_scalar(values, distance_array)

    def interaction_momentum(self, momentum):
        r"""Return $\widetilde V(k)=2gK_0(a|k|)$."""
        momentum_array = np.asarray(momentum, dtype=float)
        values = 2.0 * self.coupling * k0(
            self.softening * np.abs(momentum_array)
        )
        return _scalar_if_scalar(values, momentum_array)

    def luttinger_parameter(self, momentum):
        r"""Return the exact momentum-dependent Luttinger parameter."""
        momentum_array = np.asarray(momentum, dtype=float)
        interaction = np.asarray(
            self.interaction_momentum(momentum_array),
            dtype=float,
        )
        values = np.sqrt(
            self.fermi_velocity
            / (self.fermi_velocity + interaction / np.pi)
        )
        return _scalar_if_scalar(values, momentum_array)

    def mode_velocity(self, momentum):
        """Return the exact charge-mode velocity."""
        momentum_array = np.asarray(momentum, dtype=float)
        interaction = np.asarray(
            self.interaction_momentum(momentum_array),
            dtype=float,
        )
        values = np.sqrt(
            self.fermi_velocity
            * (self.fermi_velocity + interaction / np.pi)
        )
        return _scalar_if_scalar(values, momentum_array)

    def exact_squeezing(self, momentum):
        r"""Return $\theta_*(k)=-\frac12\log K(k)$."""
        momentum_array = np.asarray(momentum, dtype=float)
        values = -0.5 * np.log(self.luttinger_parameter(momentum_array))
        return _scalar_if_scalar(values, momentum_array)

    def static_structure_factor(self, momentum):
        """Return the exact density structure factor per unit length."""
        momentum_array = np.asarray(momentum, dtype=float)
        values = (
            np.abs(momentum_array)
            * self.luttinger_parameter(momentum_array)
            / (2.0 * np.pi)
        )
        return _scalar_if_scalar(values, momentum_array)

    def ground_state_energy_shift_density(
        self,
        *,
        momentum_max=np.inf,
        epsabs=1.0e-12,
        epsrel=1.0e-11,
    ):
        """Return the exact normal-ordered Bogoliubov energy density."""
        upper = float(momentum_max)
        if upper <= 0.0:
            raise ValueError("momentum_max must be positive.")

        def integrand(momentum):
            if momentum == 0.0:
                return 0.0
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
            limit=400,
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
        """Return the UV-regulated connected density correlation."""
        return _luttinger_density_correlation(
            self.luttinger_parameter,
            distances,
            uv_cutoff=uv_cutoff,
            points=points,
            integration_max=integration_max,
        )


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

        with np.errstate(divide="ignore"):
            zero_squeezing = float(model.exact_squeezing(0.0))
        fit_minimum = scale * (
            1.0e-4 if np.isfinite(zero_squeezing) else 1.0e-8
        )
        fit_momentum = np.geomspace(
            fit_minimum,
            scale * 1.0e2,
            500,
        )
        if np.isfinite(zero_squeezing):
            fit_momentum = np.concatenate([[0.0], fit_momentum])
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


def cmps_luttinger_spectra(
    state,
    momentum,
    *,
    contraction_backend="explicit",
    iterative_tolerance=1.0e-10,
    iterative_maxiter=None,
):
    r"""Return the even normal and anomalous spectra of a one-field cMPS.

    The local continuum field combines the two chiral sectors as
    $a_{k>0}=b_{R,k}$ and $a_{k<0}=b_{L,-k}$.  Disconnected zero-momentum
    contributions are projected out because the Luttinger Hamiltonian carries
    an overall factor $|k|$.
    """
    if not isinstance(state, ContinuousMPS) or state.num_fields != 1:
        raise ValueError("Luttinger spectra require a single-field ContinuousMPS.")
    backend = str(contraction_backend).lower().replace("-", "_")
    if backend in {"hierarchy", "heom", "hierarchy_iterative", "matrix_free"}:
        return cletta_luttinger_spectra_hierarchy(
            state,
            momentum,
            tolerance=iterative_tolerance,
            maxiter=iterative_maxiter,
        )
    if backend != "explicit":
        raise ValueError("unsupported Luttinger contraction backend.")
    momentum_array = np.asarray(momentum, dtype=float)
    if np.any(momentum_array < 0.0):
        raise ValueError("momentum must be non-negative.")

    flat_momentum = momentum_array.reshape(-1)
    poles, normal_coefficients, anomalous_coefficients = (
        _cmps_luttinger_spectral_data(state)
    )
    normal = np.empty(flat_momentum.size, dtype=float)
    anomalous = np.empty(flat_momentum.size, dtype=float)
    chunk_size = max(1, 2_000_000 // len(poles))
    for start in range(0, flat_momentum.size, chunk_size):
        stop = min(start + chunk_size, flat_momentum.size)
        selected = flat_momentum[start:stop]
        minus_denominator = (
            poles[:, np.newaxis] - 1.0j * selected[np.newaxis, :]
        )
        plus_denominator = (
            poles[:, np.newaxis] + 1.0j * selected[np.newaxis, :]
        )
        normal[start:stop] = np.real(
            -np.sum(
                normal_coefficients[:, np.newaxis] / minus_denominator,
                axis=0,
            )
            - np.sum(
                normal_coefficients[:, np.newaxis] / plus_denominator,
                axis=0,
            )
        )
        anomalous[start:stop] = np.real(
            -np.sum(
                anomalous_coefficients[:, np.newaxis] / minus_denominator,
                axis=0,
            )
            - np.sum(
                anomalous_coefficients[:, np.newaxis] / plus_denominator,
                axis=0,
            )
        )
    normal = normal.reshape(momentum_array.shape)
    anomalous = anomalous.reshape(momentum_array.shape)
    if momentum_array.ndim == 0:
        return float(normal), float(anomalous)
    return normal, anomalous


def cletta_luttinger_spectra_hierarchy(
    state,
    momentum,
    *,
    tolerance=1.0e-10,
    maxiter=None,
):
    """Return cLETTA spectra from sparse two-sided hierarchy resolvents."""
    from scipy.sparse import bmat, csc_matrix, eye as sparse_eye
    from scipy.sparse.linalg import LinearOperator, gmres, splu

    from .cletta import (
        apply_cletta_multimode_bra_insertion,
        apply_cletta_multimode_ket_insertion,
        cletta_memory_fock_keys,
        cletta_multimode_hierarchy_sparse_generator,
    )

    if not isinstance(state, ContinuousMPS) or state.num_fields != 1:
        raise ValueError("Luttinger spectra require a single-field ContinuousMPS.")
    if state.cletta_base is None or state.cletta_tie_matrices is None:
        raise ValueError("hierarchy contraction requires a cLETTA memory state.")
    momentum_array = np.asarray(momentum, dtype=float)
    if np.any(momentum_array < 0.0):
        raise ValueError("momentum must be non-negative.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")

    base = state.cletta_base
    ties = np.asarray(state.cletta_tie_matrices)
    rates = np.asarray(state.cletta_decay_rates, dtype=float)
    frequencies = state.cletta_frequencies
    if frequencies is None:
        frequencies = np.zeros_like(rates)
    else:
        frequencies = np.asarray(frequencies, dtype=float)
    depth = int(state.cletta_depth)
    keys = cletta_memory_fock_keys(len(rates), depth)
    memory_dim = len(keys)
    block_shape = (memory_dim, memory_dim, base.bond_dim, base.bond_dim)
    size = int(np.prod(block_shape))
    iteration_limit = (
        int(maxiter) if maxiter is not None else max(300, 2 * size)
    )

    generator = cletta_multimode_hierarchy_sparse_generator(
        base.q,
        base.r,
        ties,
        rates,
        depth=depth,
        frequencies=frequencies,
    ).tocsc()
    initial = np.zeros(block_shape, dtype=np.complex128)
    initial[0, 0] = np.eye(base.bond_dim, dtype=np.complex128)
    initial = initial.reshape(-1)
    if size > 1:
        probe = np.arange(1, size + 1, dtype=float)
        initial += 1.0e-6 * probe / np.linalg.norm(probe)
    eigenvalue, left, right = _dominant_sparse_biorthogonal_pair(
        generator,
        initial,
        tolerance=tolerance,
        maxiter=iteration_limit,
        label="cLETTA Luttinger hierarchy",
    )

    def ket_action(vector):
        return apply_cletta_multimode_ket_insertion(
            np.asarray(vector).reshape(block_shape),
            base.r,
            ties,
            rates,
            depth=depth,
            frequencies=frequencies,
        ).reshape(-1)

    def bra_action(vector):
        return apply_cletta_multimode_bra_insertion(
            np.asarray(vector).reshape(block_shape),
            base.r,
            ties,
            rates,
            depth=depth,
            frequencies=frequencies,
        ).reshape(-1)

    source = ket_action(right)
    source -= right * np.vdot(left, source)
    shifted = generator - eigenvalue * sparse_eye(
        size,
        dtype=np.complex128,
        format="csc",
    )
    bordered = None
    iterative_solves = size > 128
    eye_virtual = np.eye(base.bond_dim, dtype=np.complex128)
    base_transfer = (
        np.kron(base.q, eye_virtual)
        + np.kron(eye_virtual, base.q.conj())
        + np.kron(base.r, base.r.conj())
    )
    decay_ket = np.asarray(keys, dtype=float) @ (
        rates + 1.0j * frequencies
    )
    decay_bra = np.asarray(keys, dtype=float) @ (
        rates - 1.0j * frequencies
    )
    block_size = base.bond_dim**2
    block_identity = np.eye(block_size, dtype=np.complex128)

    def block_preconditioner(momentum_value):
        shifts = (
            decay_ket[:, np.newaxis]
            + decay_bra[np.newaxis, :]
            + eigenvalue
            + 1.0j * float(momentum_value)
        )
        blocks = (
            base_transfer[np.newaxis, np.newaxis, :, :]
            - shifts[:, :, np.newaxis, np.newaxis]
            * block_identity[np.newaxis, np.newaxis, :, :]
        )
        inverses = np.linalg.inv(blocks)

        def apply(vector):
            shaped = np.asarray(vector).reshape(
                memory_dim,
                memory_dim,
                block_size,
            )
            return np.einsum(
                "ijab,ijb->ija",
                inverses,
                shaped,
                optimize=True,
            ).reshape(-1)

        return LinearOperator(
            (size, size),
            matvec=apply,
            dtype=np.complex128,
        )

    def solve(momentum_value):
        nonlocal bordered
        if iterative_solves:
            if momentum_value == 0.0:
                operator = LinearOperator(
                    (size, size),
                    matvec=lambda vector: (
                        shifted @ vector
                        + right * np.vdot(left, vector)
                    ),
                    dtype=np.complex128,
                )
            else:
                operator = shifted - 1.0j * momentum_value * sparse_eye(
                    size,
                    dtype=np.complex128,
                    format="csc",
                )
            solution, info = gmres(
                operator,
                source,
                M=block_preconditioner(momentum_value),
                rtol=max(tolerance, 1.0e-10),
                atol=0.0,
                restart=min(160, size),
                maxiter=iteration_limit,
            )
            if info != 0:
                raise FloatingPointError(
                    "cLETTA Luttinger shifted GMRES did not converge "
                    f"at k={momentum_value:.8g} (info={info})."
                )
            return solution
        if momentum_value == 0.0:
            if bordered is None:
                bordered = splu(
                    bmat(
                        [
                            [shifted, csc_matrix(right[:, np.newaxis])],
                            [
                                csc_matrix(left.conj()[np.newaxis, :]),
                                csc_matrix((1, 1), dtype=np.complex128),
                            ],
                        ],
                        format="csc",
                    )
                )
            rhs = np.concatenate([source, np.zeros(1, dtype=np.complex128)])
            return bordered.solve(rhs)[:size]
        matrix = shifted - 1.0j * momentum_value * sparse_eye(
            size,
            dtype=np.complex128,
            format="csc",
        )
        return splu(matrix).solve(source)

    flat_momentum = momentum_array.reshape(-1)
    normal = np.empty(flat_momentum.size, dtype=float)
    anomalous = np.empty(flat_momentum.size, dtype=float)
    for index, value in enumerate(flat_momentum):
        minus = solve(float(value))
        plus = solve(float(-value)) if value != 0.0 else minus
        normal[index] = -float(
            np.real(np.vdot(left, bra_action(minus) + bra_action(plus)))
        )
        anomalous[index] = -float(
            np.real(np.vdot(left, ket_action(minus) + ket_action(plus)))
        )
    normal = normal.reshape(momentum_array.shape)
    anomalous = anomalous.reshape(momentum_array.shape)
    if momentum_array.ndim == 0:
        return float(normal), float(anomalous)
    return normal, anomalous


def _cmps_luttinger_spectral_data(state):
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
    normal_coefficients = (
        left_row @ bra_insertion @ eigenvectors
    ) * mode_initial
    anomalous_coefficients = (
        left_row @ ket_insertion @ eigenvectors
    ) * mode_initial
    normal_coefficients[dominant] = 0.0
    anomalous_coefficients[dominant] = 0.0
    poles = eigenvalues - leading_value
    poles[dominant] = np.inf
    return poles, normal_coefficients, anomalous_coefficients


def cmps_luttinger_energy_shift_density(
    model,
    state,
    *,
    quadrature_points=180,
    momentum_scale=None,
    contraction_backend="explicit",
    iterative_tolerance=1.0e-10,
    iterative_maxiter=None,
):
    r"""Return $\langle H\rangle/L$ for a matrix cMPS or cLETTA state."""
    momentum, weights = _infinite_momentum_quadrature(
        quadrature_points,
        model.decay_rates if momentum_scale is None else momentum_scale,
    )
    normal, anomalous = cmps_luttinger_spectra(
        state,
        momentum,
        contraction_backend=contraction_backend,
        iterative_tolerance=iterative_tolerance,
        iterative_maxiter=iterative_maxiter,
    )
    interaction = np.asarray(model.interaction_momentum(momentum), dtype=float)
    b_value = interaction / (2.0 * np.pi)
    a_value = model.fermi_velocity + b_value
    integrand = momentum * (
        2.0 * a_value * normal + 2.0 * b_value * anomalous
    ) / (2.0 * np.pi)
    return float(np.dot(weights, integrand))


def cmps_luttinger_parameter(
    state,
    momentum,
    *,
    contraction_backend="explicit",
    iterative_tolerance=1.0e-10,
    iterative_maxiter=None,
):
    r"""Return the cMPS quadrature covariance $K(k)=1+2n(k)+2m(k)$."""
    normal, anomalous = cmps_luttinger_spectra(
        state,
        momentum,
        contraction_backend=contraction_backend,
        iterative_tolerance=iterative_tolerance,
        iterative_maxiter=iterative_maxiter,
    )
    return 1.0 + 2.0 * (normal + anomalous)


def cmps_luttinger_density_correlation(
    state,
    distances,
    *,
    uv_cutoff,
    points=12000,
    integration_max=None,
    contraction_backend="explicit",
    iterative_tolerance=1.0e-10,
    iterative_maxiter=None,
):
    r"""Return the UV-regulated connected density correlation of a state.

    The same convention and exponential momentum regulator as
    :meth:`ExponentialLuttingerModel.density_correlation` are used:

    $$
    C_{nn}(x)=\frac{1}{2\pi^2}\int_0^\infty dk\,
    kK(k)e^{-k/\Lambda}\cos(kx).
    $$
    """
    return _luttinger_density_correlation(
        lambda momentum: cmps_luttinger_parameter(
            state,
            momentum,
            contraction_backend=contraction_backend,
            iterative_tolerance=iterative_tolerance,
            iterative_maxiter=iterative_maxiter,
        ),
        distances,
        uv_cutoff=uv_cutoff,
        points=points,
        integration_max=integration_max,
    )


def _luttinger_cletta_sparse_implicit_value_gradient(
    model,
    *,
    bond_dim,
    num_modes,
    depth,
    reference_rates,
    optimize_memory_rates,
    lower_rate,
    upper_rate,
    quadrature_points,
    regularization,
    tolerance,
    maxiter,
):
    """Build the Luttinger energy and its sparse implicit gradient."""
    from scipy.sparse import bmat, csc_matrix, eye as sparse_eye
    from scipy.sparse.linalg import LinearOperator, gmres, splu

    from .cletta import (
        _multimode_memory_operators,
        cletta_multimode_hierarchy_sparse_generator,
        cletta_multimode_memory_matrices,
        hierarchy_blocks_to_matrix,
        matrix_to_hierarchy_blocks,
    )

    dim = int(bond_dim)
    modes = int(num_modes)
    depth = int(depth)
    base_size = canonical_parameter_size(dim)
    tie_size = modes * dim * dim
    parameter_size = base_size + tie_size + (
        modes if optimize_memory_rates else 0
    )
    pairs = skew_pairs(dim)
    reference_rates = np.asarray(reference_rates, dtype=float)
    momentum, quadrature_weights = _infinite_momentum_quadrature(
        quadrature_points,
        model.decay_rates,
    )
    interaction = np.asarray(model.interaction_momentum(momentum), dtype=float)
    b_value = interaction / (2.0 * np.pi)
    a_value = model.fermi_velocity + b_value
    normal_weights = quadrature_weights * momentum * 2.0 * a_value / (
        2.0 * np.pi
    )
    anomalous_weights = quadrature_weights * momentum * 2.0 * b_value / (
        2.0 * np.pi
    )
    regularization = float(regularization)
    tolerance = float(tolerance)

    if modes:
        keys, _key_to_index, annihilation, number = _multimode_memory_operators(
            modes,
            depth,
            np.complex128,
        )
        memory_dim = len(keys)
    else:
        keys = np.zeros((1, 0), dtype=np.int64)
        annihilation = np.zeros((0, 1, 1), dtype=np.complex128)
        number = np.zeros_like(annihilation)
        memory_dim = 1
    block_shape = (memory_dim, memory_dim, dim, dim)
    size = int(np.prod(block_shape))
    maxiter = int(maxiter) if maxiter is not None else max(300, 2 * size)
    identity_size = sparse_eye(size, dtype=np.complex128, format="csc")
    zero_scalar = csc_matrix((1, 1), dtype=np.complex128)
    eye_memory = np.eye(memory_dim, dtype=np.complex128)
    eye_virtual = np.eye(dim, dtype=np.complex128)

    def vector_to_matrix(vector):
        return hierarchy_blocks_to_matrix(
            np.asarray(vector).reshape(block_shape)
        )

    def matrix_to_vector(matrix):
        return matrix_to_hierarchy_blocks(
            matrix,
            bond_dim=dim,
            memory_dim=memory_dim,
        ).reshape(-1)

    def ket_action(vector, operator):
        return matrix_to_vector(operator @ vector_to_matrix(vector))

    def ket_adjoint_action(vector, operator):
        return matrix_to_vector(operator.conj().T @ vector_to_matrix(vector))

    def bra_action(vector, operator):
        return matrix_to_vector(vector_to_matrix(vector) @ operator.conj().T)

    def bra_adjoint_action(vector, operator):
        return matrix_to_vector(vector_to_matrix(vector) @ operator)

    def ket_derivative_action(vector, derivative):
        return matrix_to_vector(derivative @ vector_to_matrix(vector))

    def bra_derivative_action(vector, derivative):
        return matrix_to_vector(
            vector_to_matrix(vector) @ derivative.conj().T
        )

    def transfer_derivative_action(vector, dq, dr, q_memory, r_memory):
        matrix = vector_to_matrix(vector)
        out = dq @ matrix + matrix @ dq.conj().T
        out += dr @ matrix @ r_memory.conj().T
        out += r_memory @ matrix @ dr.conj().T
        return matrix_to_vector(out)

    def transfer_adjoint_derivative_action(
        vector,
        dq,
        dr,
        q_memory,
        r_memory,
    ):
        matrix = vector_to_matrix(vector)
        out = dq.conj().T @ matrix + matrix @ dq
        out += dr.conj().T @ matrix @ r_memory
        out += r_memory.conj().T @ matrix @ dr
        return matrix_to_vector(out)

    def unpack(parameters):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.size != parameter_size:
            raise ValueError(
                f"parameter size {parameters.size} does not match "
                f"{parameter_size}."
            )
        base = ContinuousMPS.from_canonical_parameters(
            parameters[:base_size],
            dim,
        )
        offset = base_size
        ties = parameters[offset : offset + tie_size].reshape(
            modes,
            dim,
            dim,
        )
        offset += tie_size
        if optimize_memory_rates:
            rates = np.exp(
                np.clip(
                    parameters[offset : offset + modes],
                    np.log(lower_rate),
                    np.log(upper_rate),
                )
            )
        else:
            rates = reference_rates
        return base, ties, rates

    def memory_matrices(base, ties, rates):
        if modes:
            return cletta_multimode_memory_matrices(
                base.q,
                base.r,
                ties,
                rates,
                depth=depth,
            )
        return (
            np.asarray(base.q, dtype=np.complex128),
            np.asarray(base.r, dtype=np.complex128),
        )

    def sparse_generator(base, ties, rates):
        if modes:
            return cletta_multimode_hierarchy_sparse_generator(
                base.q,
                base.r,
                ties,
                rates,
                depth=depth,
            ).tocsc()
        generator = (
            np.kron(base.q, eye_virtual)
            + np.kron(eye_virtual, base.q.conj())
            + np.kron(base.r, base.r.conj())
        )
        return csc_matrix(generator)

    def parameter_derivative(index, base, rates):
        dq_base = np.zeros((dim, dim), dtype=np.complex128)
        dr_base = np.zeros((dim, dim), dtype=np.complex128)
        tie_derivatives = np.zeros(
            (modes, dim, dim),
            dtype=np.complex128,
        )
        rate_derivatives = np.zeros(modes, dtype=float)
        if index < len(pairs):
            row, column = pairs[index]
            dq_base[row, column] = 1.0
            dq_base[column, row] = -1.0
        elif index < base_size:
            r_index = index - len(pairs)
            dr_base.reshape(-1)[r_index] = 1.0
            dq_base = -0.5 * (
                dr_base.T @ base.r + base.r.T @ dr_base
            )
        elif index < base_size + tie_size:
            tie_index = index - base_size
            mode, entry = divmod(tie_index, dim * dim)
            row, column = divmod(entry, dim)
            tie_derivatives[mode, row, column] = 1.0
        else:
            mode = index - base_size - tie_size
            rate_derivatives[mode] = rates[mode]

        dq = np.kron(eye_memory, dq_base)
        dr = np.kron(eye_memory, dr_base)
        for mode in range(modes):
            dq -= rate_derivatives[mode] * np.kron(
                number[mode],
                eye_virtual,
            )
            dr += np.kron(
                annihilation[mode].conj().T,
                tie_derivatives[mode],
            )
            if rate_derivatives[mode]:
                dr += (
                    0.5
                    * rate_derivatives[mode]
                    / np.sqrt(rates[mode])
                    * np.kron(annihilation[mode], eye_virtual)
                )
        return dq, dr

    def value_gradient(parameters):
        parameters = np.asarray(parameters, dtype=float)
        base, ties, rates = unpack(parameters)
        q_memory, r_memory = memory_matrices(base, ties, rates)
        generator = sparse_generator(base, ties, rates)
        initial = np.zeros(block_shape, dtype=np.complex128)
        initial[0, 0] = np.eye(dim, dtype=np.complex128)
        initial = initial.reshape(-1)
        if size > 1:
            probe = np.arange(1, size + 1, dtype=float)
            initial += 1.0e-6 * probe / np.linalg.norm(probe)
        if size <= 2:
            dense_generator = generator.toarray()
            eigenvalues, right_vectors = np.linalg.eig(dense_generator)
            right_index = int(np.argmax(np.real(eigenvalues)))
            eigenvalue = eigenvalues[right_index]
            right = right_vectors[:, right_index]
            left_values, left_vectors = np.linalg.eig(
                dense_generator.conj().T
            )
            left_index = int(
                np.argmin(np.abs(left_values - eigenvalue.conjugate()))
            )
            left = left_vectors[:, left_index]
            overlap = np.vdot(left, right)
            if abs(overlap) <= 1.0e-12:
                raise FloatingPointError(
                    "dominant Luttinger environments are ill-conditioned."
                )
            right = right / overlap
        else:
            eigenvalue, left, right = _dominant_sparse_biorthogonal_pair(
                generator,
                initial,
                tolerance=tolerance,
                maxiter=maxiter,
                label="implicit-gradient Luttinger cLETTA",
            )
        shifted = generator - eigenvalue * identity_size
        iterative_solves = size > 128
        base_transfer = (
            np.kron(base.q, eye_virtual)
            + np.kron(eye_virtual, base.q.conj())
            + np.kron(base.r, base.r.conj())
        )
        decay_ket = np.asarray(keys, dtype=float) @ rates.astype(
            np.complex128
        )
        decay_bra = np.asarray(keys, dtype=float) @ rates.astype(
            np.complex128
        )
        block_size = dim * dim
        block_identity = np.eye(block_size, dtype=np.complex128)

        def block_preconditioner(imaginary_shift, regularizer=0.0):
            shifts = (
                decay_ket[:, np.newaxis]
                + decay_bra[np.newaxis, :]
                + eigenvalue
                - 1.0j * float(imaginary_shift)
                + float(regularizer)
            )
            blocks = (
                base_transfer[np.newaxis, np.newaxis, :, :]
                - shifts[:, :, np.newaxis, np.newaxis]
                * block_identity[np.newaxis, np.newaxis, :, :]
            )
            inverses = np.linalg.inv(blocks)
            inverse_adjoints = np.swapaxes(inverses.conj(), -1, -2)

            def apply(vector):
                shaped = np.asarray(vector).reshape(
                    memory_dim,
                    memory_dim,
                    block_size,
                )
                return np.einsum(
                    "ijab,ijb->ija",
                    inverses,
                    shaped,
                    optimize=True,
                ).reshape(-1)

            def apply_adjoint(vector):
                shaped = np.asarray(vector).reshape(
                    memory_dim,
                    memory_dim,
                    block_size,
                )
                return np.einsum(
                    "ijab,ijb->ija",
                    inverse_adjoints,
                    shaped,
                    optimize=True,
                ).reshape(-1)

            return LinearOperator(
                (size, size),
                matvec=apply,
                rmatvec=apply_adjoint,
                dtype=np.complex128,
            )

        def iterative_solve(
            matrix,
            rhs,
            preconditioner,
            *,
            rank_right=None,
            rank_left=None,
        ):
            if rank_right is None:
                operator = matrix
            else:
                operator = LinearOperator(
                    (size, size),
                    matvec=lambda vector: (
                        matrix @ vector
                        + rank_right * np.vdot(rank_left, vector)
                    ),
                    dtype=np.complex128,
                )
            solution, info = gmres(
                operator,
                rhs,
                M=preconditioner,
                rtol=max(tolerance, 1.0e-9),
                atol=0.0,
                restart=min(160, size),
                maxiter=maxiter,
            )
            if info != 0:
                raise FloatingPointError(
                    "implicit-gradient Luttinger GMRES did not converge "
                    f"(info={info})."
                )
            return solution

        if iterative_solves:
            eigen_preconditioner = block_preconditioner(
                0.0,
                regularizer=1.0e-4,
            )
        else:
            right_border = bmat(
                [
                    [shifted, -csc_matrix(right[:, np.newaxis])],
                    [csc_matrix(left.conj()[np.newaxis, :]), zero_scalar],
                ],
                format="csc",
            )
            left_border = bmat(
                [
                    [
                        shifted.conj().T,
                        -csc_matrix(left[:, np.newaxis]),
                    ],
                    [
                        csc_matrix(right.conj()[np.newaxis, :]),
                        zero_scalar,
                    ],
                ],
                format="csc",
            )
            right_factor = splu(right_border)
            left_factor = splu(left_border)

        source_unprojected = ket_action(right, r_memory)
        source_overlap = np.vdot(left, source_unprojected)
        source = source_unprojected - right * source_overlap
        gradient_left = np.zeros(size, dtype=np.complex128)
        gradient_right = np.zeros(size, dtype=np.complex128)
        eigenvalue_coefficient = 0.0j
        resolvents = []
        value = 0.0
        for momentum_value, normal_weight, anomalous_weight in zip(
            momentum,
            normal_weights,
            anomalous_weights,
        ):
            for sign in (-1.0, 1.0):
                resolvent = shifted + 1.0j * sign * momentum_value * identity_size
                if iterative_solves:
                    resolvent_preconditioner = block_preconditioner(
                        sign * momentum_value
                    )
                    solved = iterative_solve(
                        resolvent,
                        source,
                        resolvent_preconditioner,
                    )
                else:
                    factor = splu(resolvent)
                    solved = factor.solve(source)
                output = (
                    normal_weight * bra_action(solved, r_memory)
                    + anomalous_weight * ket_action(solved, r_memory)
                )
                value -= float(np.real(np.vdot(left, output)))
                gradient_left -= 0.5 * output
                gradient_solved = -0.5 * (
                    normal_weight * bra_adjoint_action(left, r_memory)
                    + anomalous_weight * ket_adjoint_action(left, r_memory)
                )
                if iterative_solves:
                    resolvent_adjoint = iterative_solve(
                        resolvent.conj().T,
                        gradient_solved,
                        resolvent_preconditioner.H,
                    )
                else:
                    resolvent_adjoint = factor.solve(
                        gradient_solved,
                        trans="H",
                    )
                overlap_right = np.vdot(resolvent_adjoint, right)
                gradient_left -= overlap_right * source_unprojected
                gradient_right += (
                    ket_adjoint_action(resolvent_adjoint, r_memory)
                    - source_overlap.conjugate() * resolvent_adjoint
                    - overlap_right.conjugate()
                    * ket_adjoint_action(left, r_memory)
                )
                eigenvalue_coefficient += np.vdot(
                    resolvent_adjoint,
                    solved,
                )
                resolvents.append(
                    (
                        float(normal_weight),
                        float(anomalous_weight),
                        solved,
                        resolvent_adjoint,
                        overlap_right,
                    )
                )

        if iterative_solves:
            right_adjoint = iterative_solve(
                shifted.conj().T,
                gradient_right
                - left * np.vdot(right, gradient_right),
                eigen_preconditioner.H,
                rank_right=left,
                rank_left=right,
            )
            left_adjoint = iterative_solve(
                shifted,
                gradient_left - right * np.vdot(left, gradient_left),
                eigen_preconditioner,
                rank_right=right,
                rank_left=left,
            )
        else:
            right_adjoint = right_factor.solve(
                np.concatenate(
                    [gradient_right, np.zeros(1, dtype=np.complex128)]
                ),
                trans="H",
            )[:size]
            left_adjoint = left_factor.solve(
                np.concatenate(
                    [gradient_left, np.zeros(1, dtype=np.complex128)]
                ),
                trans="H",
            )[:size]

        gradient = np.zeros_like(parameters)
        for parameter_index in range(parameter_size):
            dq, dr = parameter_derivative(parameter_index, base, rates)
            dg_right = transfer_derivative_action(
                right,
                dq,
                dr,
                q_memory,
                r_memory,
            )
            dg_adjoint_left = transfer_adjoint_derivative_action(
                left,
                dq,
                dr,
                q_memory,
                r_memory,
            )
            dsource_unprojected = ket_derivative_action(right, dr)
            derivative = 0.0
            for (
                normal_weight,
                anomalous_weight,
                solved,
                resolvent_adjoint,
                overlap_right,
            ) in resolvents:
                derivative -= normal_weight * float(
                    np.real(
                        np.vdot(
                            left,
                            bra_derivative_action(solved, dr),
                        )
                    )
                )
                derivative -= anomalous_weight * float(
                    np.real(
                        np.vdot(
                            left,
                            ket_derivative_action(solved, dr),
                        )
                    )
                )
                derivative += 2.0 * float(
                    np.real(
                        np.vdot(resolvent_adjoint, dsource_unprojected)
                        - overlap_right
                        * np.vdot(left, dsource_unprojected)
                    )
                )
                derivative -= 2.0 * float(
                    np.real(
                        np.vdot(
                            resolvent_adjoint,
                            transfer_derivative_action(
                                solved,
                                dq,
                                dr,
                                q_memory,
                                r_memory,
                            ),
                        )
                    )
                )
            derivative -= 2.0 * float(
                np.real(np.vdot(right_adjoint, dg_right))
            )
            derivative -= 2.0 * float(
                np.real(np.vdot(left_adjoint, dg_adjoint_left))
            )
            derivative += 2.0 * float(
                np.real(
                    eigenvalue_coefficient * np.vdot(left, dg_right)
                )
            )
            derivative += 2.0 * regularization * parameters[parameter_index]
            gradient[parameter_index] = derivative

        value += regularization * float(np.dot(parameters, parameters))
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return 1.0e30, np.zeros_like(parameters)
        return float(value), gradient

    return value_gradient


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
    contraction_backend="explicit",
    iterative_tolerance=1.0e-10,
    iterative_maxiter=None,
    gradient_backend="implicit",
):
    r"""Optimize a genuine $(D,M)$ matrix cLETTA for the Luttinger model.

    The optimized variables are the left-canonical cMPS core $(Q,R)$, the
    matrix-valued memory ties $G_\mu$, and optionally their decay rates.  The
    returned object is the explicit finite-depth cLETTA ``ContinuousMPS``.
    Sparse implicit differentiation of the dominant environments and shifted
    transfer solves is the default; set ``gradient_backend="finite_difference"``
    only for diagnostics.
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
        contraction_backend = "explicit"
    gradient_backend = str(gradient_backend).lower()
    if gradient_backend not in {"implicit", "finite_difference"}:
        raise ValueError(
            "gradient_backend must be 'implicit' or 'finite_difference'."
        )

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
            contraction_backend=contraction_backend,
            iterative_tolerance=iterative_tolerance,
            iterative_maxiter=iterative_maxiter,
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

    implicit_value_gradient = None
    if gradient_backend == "implicit":
        raw_implicit_value_gradient = _luttinger_cletta_sparse_implicit_value_gradient(
            model,
            bond_dim=bond_dim,
            num_modes=num_modes,
            depth=depth,
            reference_rates=reference_rates,
            optimize_memory_rates=optimize_memory_rates,
            lower_rate=lower_rate,
            upper_rate=upper_rate,
            quadrature_points=quadrature_points,
            regularization=regularization,
            tolerance=iterative_tolerance,
            maxiter=iterative_maxiter,
        )

        def implicit_value_gradient(parameters):
            nonlocal evaluations
            evaluations += 1
            try:
                return raw_implicit_value_gradient(parameters)
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                return 1.0e30, np.zeros_like(parameters, dtype=float)

    matrix_bounds = [(-4.0, 4.0)] * (base_size + tie_size)
    bounds = list(matrix_bounds)
    if optimize_memory_rates:
        bounds.extend(
            [(np.log(lower_rate), np.log(upper_rate))] * num_modes
        )

    results = []
    for initial in candidates:
        selected_objective = (
            implicit_value_gradient
            if implicit_value_gradient is not None
            else objective
        )
        result = minimize(
            selected_objective,
            initial,
            method="L-BFGS-B",
            jac=implicit_value_gradient is not None,
            bounds=bounds,
            options={"maxiter": int(maxiter), "maxls": 60, "ftol": 1.0e-12},
        )
        results.append(result)

    final_candidates = [
        (np.asarray(result.x, dtype=float), result.success, str(result.message))
        for result in results
    ]
    final_candidates.extend(
        (initial, False, "unoptimized matrix cLETTA seed")
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
    state.algorithm = (
        f"matrix-cletta-luttinger-{gradient_backend}-L-BFGS-B"
    )
    return state


def _scalar_if_scalar(values, argument):
    values = np.asarray(values)
    if np.asarray(argument).ndim == 0:
        return values.item()
    return values


def _luttinger_density_correlation(
    parameter,
    distances,
    *,
    uv_cutoff,
    points,
    integration_max,
):
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
        * np.asarray(parameter(momentum), dtype=float)
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
    "CoulombLuttingerModel",
    "ExponentialLuttingerModel",
    "GaussianLuttingerCLETTA",
    "cmps_luttinger_energy_shift_density",
    "cmps_luttinger_density_correlation",
    "cmps_luttinger_parameter",
    "cmps_luttinger_spectra",
    "cletta_luttinger_spectra_hierarchy",
    "optimize_luttinger_cletta",
]
