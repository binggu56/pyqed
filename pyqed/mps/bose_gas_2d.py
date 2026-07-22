"""Direct-continuum benchmarks and shell contractions for a 2D Bose gas.

The module deliberately separates three layers:

* the universal dilute-gas expansion written in terms of the physical 2D
  scattering length;
* the exactly contractible Bogoliubov Hamiltonian for a smooth finite-range
  interaction;
* a generic radial/angular hierarchical transfer contraction.

The last layer contracts finite hLETTA truncations without interpreting
radial or angular quadrature nodes as physical orbitals.  A model-specific
non-Gaussian tensor parameterization can be placed on top of that contraction
without changing the continuum benchmark conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.linalg import eig, expm, expm_frechet
from scipy.optimize import brentq, minimize
from scipy.special import i0e

from .cletta import cletta_memory_matrices


_EULER_GAMMA = 0.5772156649015328606


def _scalar_if_scalar(values, original):
    values = np.asarray(values)
    return values.item() if np.asarray(original).ndim == 0 else values


@dataclass
class RankOneDensityTransferChannel2D:
    r"""One radial momentum-transfer profile with continuous direction.

    The channel is specified by ``u(q)`` and represents

    $$
    V_1(q)=|u(q)|^2.
    $$

    For an isotropic state, all state dependence enters through
    ``S_rho(q) = <rho_q rho_-q> / area``.  The interaction energy density is

    $$
    \mathcal E_{int}=\mathcal E_{MF}+
    \frac12\int_0^{q_c}\frac{q\,dq}{2\pi}
    |u(q)|^2[S_\rho(q)-n].
    $$

    This object owns only continuum momentum-transfer quadrature.  A Gaussian
    state or an hLETTA contraction can provide ``S_rho`` through the same API.
    Quadrature nodes are evaluation points, not physical orbitals.
    """

    radial_profile: object
    momentum_cutoff: float
    radial_points: int = 64

    def __post_init__(self):
        if not callable(self.radial_profile):
            raise TypeError("radial_profile must be callable.")
        self.momentum_cutoff = float(self.momentum_cutoff)
        self.radial_points = int(self.radial_points)
        if not np.isfinite(self.momentum_cutoff) or self.momentum_cutoff <= 0.0:
            raise ValueError("momentum_cutoff must be finite and positive.")
        if self.radial_points < 8:
            raise ValueError("radial_points must be at least eight.")

    def radial_quadrature(self):
        nodes, legendre_weights = leggauss(self.radial_points)
        transformed = 0.5 * (nodes + 1.0)
        momenta = self.momentum_cutoff * transformed**2
        weights = (
            legendre_weights
            * self.momentum_cutoff**2
            * transformed**3
            / (2.0 * np.pi)
        )
        return momenta, weights

    def interaction_energy_density(
        self,
        density_structure,
        *,
        density,
        mean_field_energy_density,
    ):
        """Contract a supplied radial density structure factor."""
        momenta, weights = self.radial_quadrature()
        if callable(density_structure):
            values = np.asarray(
                [density_structure(float(q_value)) for q_value in momenta],
                dtype=np.complex128,
            )
        else:
            values = np.asarray(density_structure, dtype=np.complex128)
            if values.shape != momenta.shape:
                raise ValueError(
                    "density_structure values must match radial_points."
                )
        profiles = np.asarray(self.radial_profile(momenta), dtype=np.complex128)
        if profiles.shape != momenta.shape:
            profiles = np.broadcast_to(profiles, momenta.shape)
        correction = 0.5 * weights @ (
            np.abs(profiles) ** 2 * (values - float(density))
        )
        energy = complex(mean_field_energy_density) + correction
        return float(np.real_if_close(energy).real)

    def normal_ordered_interaction_energy_density(self, normal_structure):
        r"""Return ``0.5 integral_q |u(q)|^2 < :rho_q rho_-q: >/area``."""
        momenta, weights = self.radial_quadrature()
        if callable(normal_structure):
            values = np.asarray(
                [normal_structure(float(q_value)) for q_value in momenta],
                dtype=np.complex128,
            )
        else:
            values = np.asarray(normal_structure, dtype=np.complex128)
            if values.shape != momenta.shape:
                raise ValueError(
                    "normal_structure values must match radial_points."
                )
        profiles = np.asarray(self.radial_profile(momenta), dtype=np.complex128)
        if profiles.shape != momenta.shape:
            profiles = np.broadcast_to(profiles, momenta.shape)
        energy = 0.5 * weights @ (np.abs(profiles) ** 2 * values)
        if abs(energy.imag) > 1.0e-8 * max(1.0, abs(energy.real)):
            raise FloatingPointError(
                "normal-ordered interaction energy acquired an imaginary part."
            )
        return float(energy.real)


@dataclass
class DiluteBoseGas2D:
    r"""Universal thermodynamic energy of a dilute repulsive 2D Bose gas.

    Let ``density`` be the particle density and ``scattering_length`` the 2D
    two-body scattering length.  With

    $$
    Y=|\log(n a_{2D}^2)|^{-1},
    $$

    the energy density through the universal constant-order correction is

    $$
    \mathcal E=
    4\pi\frac{\hbar^2}{2m}n^2Y
    \left[1-Y|\log Y|+
    \left(2\gamma+\frac12+\log\pi\right)Y\right].
    $$

    ``kinetic_prefactor`` is ``hbar**2 / (2 m)``.  The formula is a dilute
    asymptotic expansion, not an exact equation of state at arbitrary gas
    parameter.
    """

    density: float
    scattering_length: float
    kinetic_prefactor: float = 1.0

    def __post_init__(self):
        self.density = float(self.density)
        self.scattering_length = float(self.scattering_length)
        self.kinetic_prefactor = float(self.kinetic_prefactor)
        if self.density <= 0.0:
            raise ValueError("density must be positive.")
        if self.scattering_length <= 0.0:
            raise ValueError("scattering_length must be positive.")
        if self.kinetic_prefactor <= 0.0:
            raise ValueError("kinetic_prefactor must be positive.")
        if not (0.0 < self.gas_parameter < 1.0):
            raise ValueError(
                "the dilute expansion requires "
                "0 < density * scattering_length**2 < 1."
            )

    @property
    def gas_parameter(self) -> float:
        return self.density * self.scattering_length**2

    @property
    def expansion_parameter(self) -> float:
        return 1.0 / abs(np.log(self.gas_parameter))

    @property
    def universal_constant(self) -> float:
        return 2.0 * _EULER_GAMMA + 0.5 + np.log(np.pi)

    @property
    def leading_energy_density(self) -> float:
        return float(
            4.0
            * np.pi
            * self.kinetic_prefactor
            * self.density**2
            * self.expansion_parameter
        )

    @property
    def logarithmic_energy_density(self) -> float:
        y_value = self.expansion_parameter
        return float(
            self.leading_energy_density
            * (1.0 - y_value * abs(np.log(y_value)))
        )

    @property
    def energy_density(self) -> float:
        """Return the expansion through relative order ``Y``."""
        y_value = self.expansion_parameter
        return float(
            self.leading_energy_density
            * (
                1.0
                - y_value * abs(np.log(y_value))
                + self.universal_constant * y_value
            )
        )

    @property
    def constant_order_energy_correction(self) -> float:
        r"""Return the universal ``Y**2`` constant-order contribution.

        This term belongs to the rigorously established 2D LHY-order energy
        expansion.  It is not the still higher-order correction customarily
        called the first correction beyond Bogoliubov theory.
        """
        return self.energy_density - self.logarithmic_energy_density


@dataclass
class GaussianPotentialBoseGas2D:
    r"""Bogoliubov reference for a smooth finite-range 2D interaction.

    The momentum-space potential is

    $$
    \widetilde V(k)=g\exp(-\sigma^2k^2/2),
    $$

    and the free dispersion is ``epsilon(k) = kinetic_prefactor * k**2``.
    The Gaussian range makes all Bogoliubov radial integrals ultraviolet
    finite without introducing a momentum grid or finite box.
    """

    density: float
    interaction_strength: float
    interaction_range: float
    kinetic_prefactor: float = 1.0

    def __post_init__(self):
        self.density = float(self.density)
        self.interaction_strength = float(self.interaction_strength)
        self.interaction_range = float(self.interaction_range)
        self.kinetic_prefactor = float(self.kinetic_prefactor)
        if self.density <= 0.0:
            raise ValueError("density must be positive.")
        if self.interaction_strength <= 0.0:
            raise ValueError("interaction_strength must be positive.")
        if self.interaction_range <= 0.0:
            raise ValueError("interaction_range must be positive.")
        if self.kinetic_prefactor <= 0.0:
            raise ValueError("kinetic_prefactor must be positive.")

    def free_dispersion(self, momentum):
        momentum = np.asarray(momentum, dtype=float)
        if np.any(momentum < 0.0):
            raise ValueError("radial momentum must be non-negative.")
        values = self.kinetic_prefactor * momentum**2
        return _scalar_if_scalar(values, momentum)

    def interaction_momentum(self, momentum):
        momentum = np.asarray(momentum, dtype=float)
        if np.any(momentum < 0.0):
            raise ValueError("radial momentum must be non-negative.")
        values = self.interaction_strength * np.exp(
            -0.5 * (self.interaction_range * momentum) ** 2
        )
        return _scalar_if_scalar(values, momentum)

    def density_transfer_profile(self, momentum):
        r"""Return the single radial channel ``u(q)`` with ``V(q)=|u(q)|^2``."""
        momentum = np.asarray(momentum, dtype=float)
        if np.any(momentum < 0.0):
            raise ValueError("radial momentum must be non-negative.")
        values = np.sqrt(self.interaction_strength) * np.exp(
            -0.25 * (self.interaction_range * momentum) ** 2
        )
        return _scalar_if_scalar(values, momentum)

    def quasiparticle_dispersion(self, momentum):
        momentum_array = np.asarray(momentum, dtype=float)
        epsilon = np.asarray(self.free_dispersion(momentum_array))
        interaction = self.density * np.asarray(
            self.interaction_momentum(momentum_array)
        )
        values = np.sqrt(epsilon * (epsilon + 2.0 * interaction))
        return _scalar_if_scalar(values, momentum_array)

    def static_structure_factor(self, momentum):
        r"""Return the Bogoliubov static structure factor ``epsilon / E``."""
        momentum_array = np.asarray(momentum, dtype=float)
        epsilon = np.asarray(self.free_dispersion(momentum_array))
        dispersion = np.asarray(self.quasiparticle_dispersion(momentum_array))
        values = np.divide(
            epsilon,
            dispersion,
            out=np.zeros_like(epsilon),
            where=dispersion > 0.0,
        )
        return _scalar_if_scalar(values, momentum_array)

    def squeezing(self, momentum):
        r"""Return the positive antipodal-pair squeezing ``r(k)``."""
        momentum_array = np.asarray(momentum, dtype=float)
        epsilon = np.asarray(self.free_dispersion(momentum_array))
        interaction = self.density * np.asarray(
            self.interaction_momentum(momentum_array)
        )
        ratio = np.divide(
            2.0 * interaction,
            epsilon,
            out=np.full_like(epsilon, np.inf),
            where=epsilon > 0.0,
        )
        values = 0.25 * np.log1p(ratio)
        return _scalar_if_scalar(values, momentum_array)

    @property
    def mean_field_energy_density(self) -> float:
        return 0.5 * self.interaction_strength * self.density**2

    def bogoliubov_energy_correction_density(
        self,
        *,
        momentum_max=np.inf,
        epsabs=1.0e-11,
        epsrel=1.0e-10,
    ) -> float:
        r"""Return the UV-finite zero-point correction per area.

        The stable radial integral is

        $$
        \Delta\mathcal E=
        -\frac{1}{4\pi}\int_0^\infty dk\,k
        \frac{B_k^2}{E_k+A_k},
        $$

        where ``A_k = epsilon_k + n V_k`` and ``B_k = n V_k``.
        """
        upper = float(momentum_max)
        if upper <= 0.0:
            raise ValueError("momentum_max must be positive.")

        def integrand(momentum):
            epsilon = float(self.free_dispersion(momentum))
            b_value = self.density * float(self.interaction_momentum(momentum))
            a_value = epsilon + b_value
            dispersion = np.sqrt(epsilon * (epsilon + 2.0 * b_value))
            return (
                -momentum
                * b_value**2
                / (4.0 * np.pi * (dispersion + a_value))
            )

        value, _ = quad(
            integrand,
            0.0,
            upper,
            epsabs=float(epsabs),
            epsrel=float(epsrel),
            limit=400,
        )
        return float(value)

    @property
    def bogoliubov_energy_density(self) -> float:
        return self.mean_field_energy_density + self.bogoliubov_energy_correction_density()

    def depletion_density(
        self,
        *,
        momentum_max=np.inf,
        epsabs=1.0e-11,
        epsrel=1.0e-10,
    ) -> float:
        """Return the direct-continuum Bogoliubov depletion density."""
        upper = float(momentum_max)
        if upper <= 0.0:
            raise ValueError("momentum_max must be positive.")

        def integrand(momentum):
            epsilon = float(self.free_dispersion(momentum))
            b_value = self.density * float(self.interaction_momentum(momentum))
            a_value = epsilon + b_value
            dispersion = np.sqrt(epsilon * (epsilon + 2.0 * b_value))
            if dispersion == 0.0:
                return 0.0
            occupation = b_value**2 / (
                2.0 * dispersion * (a_value + dispersion)
            )
            return momentum * occupation / (2.0 * np.pi)

        value, _ = quad(
            integrand,
            0.0,
            upper,
            epsabs=float(epsabs),
            epsrel=float(epsrel),
            limit=400,
        )
        return float(value)

    def quadratic_energy_density_for_squeezing(
        self,
        squeezing,
        *,
        momentum_max=np.inf,
        epsabs=1.0e-10,
        epsrel=1.0e-9,
    ) -> float:
        r"""Evaluate the quadratic Hamiltonian for a trial ``r(k)``.

        This is the variational Gaussian functional

        $$
        \mathcal E[r]=\mathcal E_{MF}+
        \frac{1}{2\pi}\int_0^\infty dk\,k
        \left[A_k\sinh^2r_k-\frac{B_k}{2}\sinh(2r_k)\right].
        $$
        """
        upper = float(momentum_max)
        if upper <= 0.0:
            raise ValueError("momentum_max must be positive.")

        def integrand(momentum):
            epsilon = float(self.free_dispersion(momentum))
            b_value = self.density * float(self.interaction_momentum(momentum))
            a_value = epsilon + b_value
            squeeze = float(squeezing(momentum))
            return momentum * (
                a_value * np.sinh(squeeze) ** 2
                - 0.5 * b_value * np.sinh(2.0 * squeeze)
            ) / (2.0 * np.pi)

        value, _ = quad(
            integrand,
            0.0,
            upper,
            epsabs=float(epsabs),
            epsrel=float(epsrel),
            limit=400,
        )
        return float(self.mean_field_energy_density + value)

    def full_gaussian_energy_density_for_squeezing(
        self,
        squeezing,
        *,
        momentum_max=None,
        quadrature_points=96,
    ) -> float:
        r"""Evaluate the full interacting Hamiltonian in a Gaussian state.

        The trial state has a real condensate density ``n0`` and a pure,
        isotropic squeezed fluctuation field with

        $$
        n_k=\sinh^2 r_k,\qquad
        m_k=-\sinh r_k\cosh r_k,
        $$

        where the minus sign chooses the energy-lowering anomalous phase for
        a repulsive interaction.  The physical density is fixed by

        $$
        n_0=n-\int\frac{d^2k}{(2\pi)^2}n_k.
        $$

        Wick contraction of the *full* two-body Hamiltonian gives

        $$
        \begin{aligned}
        \mathcal E_G={}&\int_k\epsilon_k n_k+\frac12V(0)n^2
        +n_0\int_kV(k)(n_k+m_k)\\
        &+\frac12\int_k\int_pV(|k-p|)
        (n_kn_p+m_km_p).
        \end{aligned}
        $$

        For the Gaussian potential the relative-angle integral is analytic;
        only direct radial continuum quadrature remains.  Quadrature nodes are
        integration points, not physical momentum orbitals.
        """
        quadrature_points = int(quadrature_points)
        if quadrature_points < 8:
            raise ValueError("quadrature_points must be at least eight.")
        if momentum_max is None:
            interaction_scale = np.sqrt(
                self.density
                * self.interaction_strength
                / self.kinetic_prefactor
            )
            momentum_max = max(
                8.0 / self.interaction_range,
                6.0 * interaction_scale,
            )
        momentum_max = float(momentum_max)
        if not np.isfinite(momentum_max) or momentum_max <= 0.0:
            raise ValueError("momentum_max must be finite and positive.")

        nodes, weights = leggauss(quadrature_points)
        transformed = 0.5 * (nodes + 1.0)
        momenta = momentum_max * transformed**2
        # k = momentum_max * t**2 resolves the logarithmic Bogoliubov
        # squeezing at k=0 while retaining direct continuum quadrature.
        radial_weights = (
            weights
            * momentum_max**2
            * transformed**3
            / (2.0 * np.pi)
        )
        squeeze = np.asarray(squeezing(momenta), dtype=float)
        if squeeze.shape != momenta.shape:
            try:
                squeeze = np.broadcast_to(squeeze, momenta.shape)
            except ValueError as error:
                raise ValueError(
                    "squeezing must return values compatible with momentum."
                ) from error
        if np.any(~np.isfinite(squeeze)):
            raise ValueError("squeezing returned non-finite values.")

        normal = np.sinh(squeeze) ** 2
        anomalous = -np.sinh(squeeze) * np.cosh(squeeze)
        depletion = float(radial_weights @ normal)
        condensate_density = self.density - depletion
        if condensate_density < -1.0e-12 * self.density:
            return float(np.inf)
        condensate_density = max(0.0, condensate_density)

        dispersion = self.kinetic_prefactor * momenta**2
        potential = self.interaction_strength * np.exp(
            -0.5 * (self.interaction_range * momenta) ** 2
        )
        kinetic = float(radial_weights @ (dispersion * normal))
        condensate_fluctuation = float(
            condensate_density
            * (radial_weights @ (potential * (normal + anomalous)))
        )

        first = momenta[:, None]
        second = momenta[None, :]
        argument = self.interaction_range**2 * first * second
        angular_average = (
            self.interaction_strength
            * np.exp(
                -0.5
                * self.interaction_range**2
                * (first - second) ** 2
            )
            * i0e(argument)
        )
        pair_density = (
            normal[:, None] * normal[None, :]
            + anomalous[:, None] * anomalous[None, :]
        )
        double_fluctuation = 0.5 * float(
            radial_weights
            @ (angular_average * pair_density)
            @ radial_weights
        )
        return float(
            kinetic
            + self.mean_field_energy_density
            + condensate_fluctuation
            + double_fluctuation
        )

    def density_transfer_energy_density_for_squeezing(
        self,
        squeezing,
        *,
        momentum_max=None,
        radial_points=64,
        angular_points=48,
    ) -> float:
        r"""Evaluate the same full Gaussian energy through density modes.

        With

        $$
        \rho_{\mathbf q}=\int_{\mathbf k}
        a^\dagger_{\mathbf{k+q}}a_{\mathbf k},
        $$

        the momentum-conserving interaction is

        $$
        \mathcal E_{int}=\frac12V(0)n^2+
        \frac12\int_{\mathbf q}V(q)
        \left[S_\rho(q)-n\right],
        $$

        where ``S_rho(q) = <rho_q rho_-q> / area``.  For a homogeneous pure
        Gaussian state and nonzero ``q``, Wick's theorem gives

        $$
        \begin{aligned}
        S_\rho(q)={}&n_0[1+2n_q+2m_q]\\
        &+\int_{\mathbf k}
        [n_{|\mathbf{k+q}|}(1+n_k)+m_{|\mathbf{k+q}|}m_k].
        \end{aligned}
        $$

        This route keeps the transferred vector ``q`` explicit and is the
        reference formula for a density-transfer hLETTA memory channel.  It
        must agree with :meth:`full_gaussian_energy_density_for_squeezing`.
        """
        radial_points = int(radial_points)
        angular_points = int(angular_points)
        if radial_points < 8 or angular_points < 8:
            raise ValueError("radial_points and angular_points must be at least eight.")
        if momentum_max is None:
            interaction_scale = np.sqrt(
                self.density
                * self.interaction_strength
                / self.kinetic_prefactor
            )
            momentum_max = max(
                8.0 / self.interaction_range,
                6.0 * interaction_scale,
            )
        momentum_max = float(momentum_max)
        if not np.isfinite(momentum_max) or momentum_max <= 0.0:
            raise ValueError("momentum_max must be finite and positive.")

        radial_nodes, radial_legendre_weights = leggauss(radial_points)
        transformed = 0.5 * (radial_nodes + 1.0)
        momenta = momentum_max * transformed**2
        radial_weights = (
            radial_legendre_weights
            * momentum_max**2
            * transformed**3
            / (2.0 * np.pi)
        )
        angles, angular_legendre_weights = leggauss(angular_points)
        relative_angles = np.pi * (angles + 1.0)
        angular_average_weights = 0.5 * angular_legendre_weights
        cosines = np.cos(relative_angles)

        squeeze = np.asarray(squeezing(momenta), dtype=float)
        if squeeze.shape != momenta.shape:
            try:
                squeeze = np.broadcast_to(squeeze, momenta.shape)
            except ValueError as error:
                raise ValueError(
                    "squeezing must return values compatible with momentum."
                ) from error
        normal = np.sinh(squeeze) ** 2
        anomalous = -np.sinh(squeeze) * np.cosh(squeeze)
        depletion = float(radial_weights @ normal)
        condensate_density = self.density - depletion
        if condensate_density < -1.0e-12 * self.density:
            return float(np.inf)
        condensate_density = max(0.0, condensate_density)

        kinetic = float(
            radial_weights
            @ (self.kinetic_prefactor * momenta**2 * normal)
        )
        density_structures = []
        for momentum_transfer in momenta:
            transfer_squeeze = float(squeezing(momentum_transfer))
            transfer_normal = np.sinh(transfer_squeeze) ** 2
            transfer_anomalous = (
                -np.sinh(transfer_squeeze) * np.cosh(transfer_squeeze)
            )
            shifted_momenta = np.sqrt(
                momenta[:, None] ** 2
                + momentum_transfer**2
                + 2.0
                * momenta[:, None]
                * momentum_transfer
                * cosines[None, :]
            )
            shifted_squeeze = np.asarray(
                squeezing(shifted_momenta), dtype=float
            )
            shifted_normal = np.sinh(shifted_squeeze) ** 2
            shifted_anomalous = (
                -np.sinh(shifted_squeeze) * np.cosh(shifted_squeeze)
            )
            convolution_by_k = (
                shifted_normal * (1.0 + normal[:, None])
                + shifted_anomalous * anomalous[:, None]
            ) @ angular_average_weights
            density_structure = (
                condensate_density
                * (1.0 + 2.0 * transfer_normal + 2.0 * transfer_anomalous)
                + radial_weights @ convolution_by_k
            )
            density_structures.append(density_structure)

        channel = RankOneDensityTransferChannel2D(
            radial_profile=self.density_transfer_profile,
            momentum_cutoff=momentum_max,
            radial_points=radial_points,
        )
        interaction = channel.interaction_energy_density(
            density_structures,
            density=self.density,
            mean_field_energy_density=self.mean_field_energy_density,
        )
        return float(kinetic + interaction)

    def optimize_full_gaussian(
        self,
        *,
        momentum_max=None,
        quadrature_points=96,
        maxiter=200,
    ):
        r"""Optimize a two-scale full-Hamiltonian Gaussian ansatz.

        The radial squeezing is

        $$
        r_{a,b}(k)=a\,r_{Bog}(b k),\qquad a,b>0.
        $$

        The method populates ``full_gaussian_energy_density``,
        ``squeezing_amplitude``, ``squeezing_momentum_scale``, ``success``, and
        ``message`` on this model and returns ``self``.
        """
        quadrature_points = int(quadrature_points)

        def trial(parameters):
            amplitude = np.exp(float(parameters[0]))
            momentum_scale = np.exp(float(parameters[1]))

            def squeezing(momentum):
                return amplitude * self.squeezing(momentum_scale * momentum)

            return self.full_gaussian_energy_density_for_squeezing(
                squeezing,
                momentum_max=momentum_max,
                quadrature_points=quadrature_points,
            )

        result = minimize(
            trial,
            np.zeros(2),
            method="Nelder-Mead",
            options={"maxiter": int(maxiter), "xatol": 1.0e-7, "fatol": 1.0e-10},
        )
        self.squeezing_amplitude = float(np.exp(result.x[0]))
        self.squeezing_momentum_scale = float(np.exp(result.x[1]))
        self.full_gaussian_energy_density = float(result.fun)
        self.success = bool(result.success and np.isfinite(result.fun))
        self.message = str(result.message)
        return self

    def optimized_full_gaussian_squeezing(self, momentum):
        """Evaluate the optimized squeezing after ``optimize_full_gaussian``."""
        if not hasattr(self, "squeezing_amplitude"):
            raise RuntimeError("call optimize_full_gaussian first.")
        return self.squeezing_amplitude * self.squeezing(
            self.squeezing_momentum_scale * np.asarray(momentum, dtype=float)
        )


@dataclass
class HierarchicalShellContraction:
    r"""Nested angular-then-radial contraction for a finite hLETTA hierarchy.

    ``angular_generator(E, theta, dE)`` must return the double-layer angular
    generator for a radial cell of width ``dE``.  Its output acts on the one
    shared outer virtual-memory environment.  The callback owns the continuum
    scaling; for an area-local generator it normally returns ``dE * L``.

    Angular quadrature produces a shell map, and radial quadrature composes
    those maps in increasing energy order.  The nodes are integration points,
    not momentum orbitals.
    """

    energy_cutoff: float
    radial_points: int = 32
    angular_points: int = 32

    def __post_init__(self):
        self.energy_cutoff = float(self.energy_cutoff)
        self.radial_points = int(self.radial_points)
        self.angular_points = int(self.angular_points)
        if self.energy_cutoff <= 0.0:
            raise ValueError("energy_cutoff must be positive.")
        if self.radial_points < 2 or self.angular_points < 2:
            raise ValueError("radial_points and angular_points must be at least two.")

    def radial_quadrature(self):
        nodes, weights = leggauss(self.radial_points)
        energies = 0.5 * self.energy_cutoff * (nodes + 1.0)
        widths = 0.5 * self.energy_cutoff * weights
        return energies, widths

    def angular_quadrature(self):
        nodes, weights = leggauss(self.angular_points)
        angles = np.pi * (nodes + 1.0)
        widths = np.pi * weights
        return angles, widths

    def shell_map(self, energy, radial_width, angular_generator):
        r"""Return ``P_theta exp integral dtheta L(E,theta,dE)``."""
        angles, angular_widths = self.angular_quadrature()
        result = None
        for angle, angular_width in zip(angles, angular_widths):
            generator = np.asarray(
                angular_generator(float(energy), float(angle), float(radial_width)),
                dtype=np.complex128,
            )
            if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
                raise ValueError("angular_generator must return a square matrix.")
            if result is None:
                result = np.eye(generator.shape[0], dtype=np.complex128)
            elif generator.shape != result.shape:
                raise ValueError("angular_generator changed dimension inside a shell.")
            result = expm(float(angular_width) * generator) @ result
        return result

    def contract(self, angular_generator, *, left_boundary, right_boundary):
        """Contract all shells and return ``(scalar, final_environment)``."""
        left = np.asarray(left_boundary, dtype=np.complex128).reshape(-1)
        environment = np.asarray(right_boundary, dtype=np.complex128).reshape(-1)
        if left.shape != environment.shape:
            raise ValueError("left and right boundaries must have the same shape.")
        energies, radial_widths = self.radial_quadrature()
        for energy, radial_width in zip(energies, radial_widths):
            shell = self.shell_map(energy, radial_width, angular_generator)
            if shell.shape != (environment.size, environment.size):
                raise ValueError("shell map dimension does not match the boundaries.")
            environment = shell @ environment
        return complex(np.vdot(left, environment)), environment

    def contract_with_generator_derivative(
        self,
        angular_generator,
        angular_derivative,
        *,
        left_boundary,
        right_boundary,
    ):
        r"""Contract a transfer product and its exact first derivative.

        ``angular_derivative(E, theta, dE)`` is the derivative of the local
        generator with respect to one scalar source.  Each matrix exponential
        is differentiated with a Fréchet derivative, so this evaluates an
        integrated one-body insertion without a finite-difference step.

        Returns ``(scalar, derivative, final_environment)``.  Dividing the
        derivative by ``scalar`` gives the normalized expectation associated
        with a logarithmic source.
        """
        left = np.asarray(left_boundary, dtype=np.complex128).reshape(-1)
        environment = np.asarray(right_boundary, dtype=np.complex128).reshape(-1)
        tangent = np.zeros_like(environment)
        if left.shape != environment.shape:
            raise ValueError("left and right boundaries must have the same shape.")

        energies, radial_widths = self.radial_quadrature()
        angles, angular_widths = self.angular_quadrature()
        for energy, radial_width in zip(energies, radial_widths):
            for angle, angular_width in zip(angles, angular_widths):
                generator = np.asarray(
                    angular_generator(
                        float(energy), float(angle), float(radial_width)
                    ),
                    dtype=np.complex128,
                )
                derivative = np.asarray(
                    angular_derivative(
                        float(energy), float(angle), float(radial_width)
                    ),
                    dtype=np.complex128,
                )
                if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
                    raise ValueError("angular_generator must return a square matrix.")
                if derivative.shape != generator.shape:
                    raise ValueError(
                        "angular_derivative must match the generator shape."
                    )
                if generator.shape != (environment.size, environment.size):
                    raise ValueError(
                        "generator dimension does not match the boundaries."
                    )
                propagator, propagator_derivative = expm_frechet(
                    float(angular_width) * generator,
                    float(angular_width) * derivative,
                    compute_expm=True,
                )
                tangent = (
                    propagator_derivative @ environment
                    + propagator @ tangent
                )
                environment = propagator @ environment

        scalar = complex(np.vdot(left, environment))
        scalar_derivative = complex(np.vdot(left, tangent))
        return scalar, scalar_derivative, environment


@dataclass
class D2M1HierarchicalCLETTA2D:
    r"""Flattened shell-ordered 2D cLETTA with ``D=2`` and one memory mode.

    The state is defined directly above the physical field vacuum by

    $$
    |\Psi\rangle = \langle l|\mathcal P_E\mathcal P_\theta
    \exp\!\left\{\int dE\,d\theta\,
    \left[Q_c+f(E)e^{i\ell\theta}R_c
    \psi^\dagger(E,\theta)\right]\right\}|r\rangle|\Omega\rangle.
    $$

    ``Q``, ``R``, and ``S`` are genuine ``2 x 2`` outer-virtual matrices.
    A single auxiliary oscillator, truncated to occupations zero and one,
    gives

    $$
    Q_c=I_m\otimes Q-\kappa N_m\otimes I_D,
    \qquad
    R_c=I_m\otimes R+\sqrt\kappa\,a_m\otimes I_D
        +a_m^\dagger\otimes S.
    $$

    Thus the ket auxiliary dimension is ``D * memory_dim = 4`` and the norm
    transfer dimension is ``16``.  ``M=1`` means one memory channel; the
    two-level memory cutoff is a separate controlled truncation.

    This class contracts the finite ansatz exactly and supplies useful field
    insertion diagnostics.  Because all auxiliary evolution is scaled by the
    radial cell width, it is the flattened lexicographic construction.  Use
    :class:`D2M1NestedCLETTA2D` when the angular memory must evolve within each
    shell before the outer radial composition.

    It does not claim
    that a particular choice of these matrices minimizes the full 2D Bose-gas
    Hamiltonian; observable insertions and that variational optimization sit
    on top of this contraction.
    """

    contraction: HierarchicalShellContraction
    q_matrix: np.ndarray
    r_matrix: np.ndarray
    tie_matrix: np.ndarray
    memory_decay: float
    radial_decay: float = 0.0
    angular_momentum: int = 0

    def __post_init__(self):
        self.q_matrix = np.asarray(self.q_matrix, dtype=np.complex128)
        self.r_matrix = np.asarray(self.r_matrix, dtype=np.complex128)
        self.tie_matrix = np.asarray(self.tie_matrix, dtype=np.complex128)
        expected = (2, 2)
        if self.q_matrix.shape != expected:
            raise ValueError("q_matrix must have shape (2, 2) for D=2.")
        if self.r_matrix.shape != expected:
            raise ValueError("r_matrix must have shape (2, 2) for D=2.")
        if self.tie_matrix.shape != expected:
            raise ValueError("tie_matrix must have shape (2, 2) for D=2.")
        self.memory_decay = float(self.memory_decay)
        self.radial_decay = float(self.radial_decay)
        self.angular_momentum = int(self.angular_momentum)
        if not np.isfinite(self.memory_decay) or self.memory_decay <= 0.0:
            raise ValueError("memory_decay must be finite and positive.")
        if not np.isfinite(self.radial_decay) or self.radial_decay < 0.0:
            raise ValueError("radial_decay must be finite and non-negative.")

    @property
    def bond_dim(self) -> int:
        return 2

    @property
    def num_memory_modes(self) -> int:
        return 1

    @property
    def memory_dim(self) -> int:
        return 2

    @property
    def effective_bond_dim(self) -> int:
        return self.bond_dim * self.memory_dim

    @property
    def transfer_dim(self) -> int:
        return self.effective_bond_dim**2

    def combined_matrices(self):
        """Return the explicit ``4 x 4`` matrices ``(Q_c, R_c)``."""
        return cletta_memory_matrices(
            self.q_matrix,
            self.r_matrix,
            self.tie_matrix,
            self.memory_decay,
            memory_dim=self.memory_dim,
        )

    def radial_envelope(self, energy):
        """Return ``f(E)=exp(-radial_decay * E / E_cutoff)``."""
        scaled_energy = float(energy) / self.contraction.energy_cutoff
        return float(np.exp(-self.radial_decay * scaled_energy))

    def physical_insertion(self, energy, theta):
        r"""Return ``f(E) exp(i ell theta) R_c``."""
        _, r_combined = self.combined_matrices()
        phase = np.exp(1.0j * self.angular_momentum * float(theta))
        return self.radial_envelope(energy) * phase * r_combined

    def angular_generator(self, energy, theta, radial_width):
        r"""Return the exact double-layer generator for one area cell.

        The factor ``radial_width`` is the outer integration measure.  The
        angular Gauss weight is supplied by :class:`HierarchicalShellContraction`.
        """
        q_combined, _ = self.combined_matrices()
        insertion = self.physical_insertion(energy, theta)
        identity = np.eye(self.effective_bond_dim, dtype=np.complex128)
        transfer = (
            np.kron(q_combined, identity)
            + np.kron(identity, q_combined.conj())
            + np.kron(insertion, insertion.conj())
        )
        return float(radial_width) * transfer

    def occupation_generator(self, energy, theta, radial_width, *, weight=1.0):
        r"""Return the source derivative for ``weight * psi^dagger psi``."""
        insertion = self.physical_insertion(energy, theta)
        jump = np.kron(insertion, insertion.conj())
        return float(radial_width) * complex(weight) * jump

    def boundary_vectors(self):
        r"""Return double-layer memory/virtual-vacuum boundary vectors."""
        ket_boundary = np.zeros(self.effective_bond_dim, dtype=np.complex128)
        ket_boundary[0] = 1.0
        double_boundary = np.kron(ket_boundary, ket_boundary.conj())
        return double_boundary.copy(), double_boundary

    def norm(self):
        """Contract and return the norm of the finite hLETTA state."""
        left, right = self.boundary_vectors()
        value, _ = self.contraction.contract(
            self.angular_generator,
            left_boundary=left,
            right_boundary=right,
        )
        return float(np.real_if_close(value).real)

    def additive_one_body_expectation(self, weight):
        r"""Return ``integral dE dtheta weight(E) <psi^dagger psi>``.

        The result is evaluated as an exact derivative of the discretized
        ordered exponential.  ``weight`` may be a callable of energy or a
        scalar.
        """
        if callable(weight):
            weight_function = weight
        else:
            constant = complex(weight)

            def weight_function(_energy):
                return constant

        left, right = self.boundary_vectors()

        def derivative(energy, theta, radial_width):
            return self.occupation_generator(
                energy,
                theta,
                radial_width,
                weight=weight_function(energy),
            )

        norm, source_derivative, _ = (
            self.contraction.contract_with_generator_derivative(
                self.angular_generator,
                derivative,
                left_boundary=left,
                right_boundary=right,
            )
        )
        if abs(norm) <= np.finfo(float).tiny:
            raise FloatingPointError("the hLETTA norm is numerically zero.")
        return source_derivative / norm

    def particle_number(self):
        r"""Return ``integral dE dtheta <psi^dagger psi>``."""
        value = self.additive_one_body_expectation(1.0)
        return float(np.real_if_close(value).real)

    def kinetic_energy(self):
        r"""Return ``integral dE dtheta E <psi^dagger psi>``."""
        value = self.additive_one_body_expectation(lambda energy: energy)
        return float(np.real_if_close(value).real)

    def _radial_shell_data(self):
        """Return ordered shell maps and right/left environments."""
        energies, radial_widths = self.contraction.radial_quadrature()
        q_combined, _ = self.combined_matrices()
        identity = np.eye(self.effective_bond_dim, dtype=np.complex128)
        shell_generators = []
        shell_maps = []
        for energy, radial_width in zip(energies, radial_widths):
            insertion = self.physical_insertion(float(energy), 0.0)
            transfer = (
                np.kron(q_combined, identity)
                + np.kron(identity, q_combined.conj())
                + np.kron(insertion, insertion.conj())
            )
            generator = float(radial_width) * transfer
            shell_generators.append(generator)
            shell_maps.append(expm(2.0 * np.pi * generator))

        left, right = self.boundary_vectors()
        right_before = []
        environment = right
        for shell in shell_maps:
            right_before.append(environment)
            environment = shell @ environment

        left_after = [None] * len(shell_maps)
        covector = left
        for index in range(len(shell_maps) - 1, -1, -1):
            left_after[index] = covector
            covector = shell_maps[index].conj().T @ covector

        norm = complex(np.vdot(left, environment))
        return (
            energies,
            radial_widths,
            shell_generators,
            right_before,
            left_after,
            norm,
        )

    def field_correlation(self, insertions):
        r"""Contract an ordered normal-field correlation.

        Each insertion is ``(radial_index, theta, operator)``, where
        ``operator`` is ``"annihilation"``, ``"creation"``, or ``"density"``.
        ``radial_index`` selects a radial quadrature evaluation point and is
        not a physical orbital label.  Insertions may be supplied in any
        order; the lexicographic ``(radial_index, theta)`` ordering is applied
        internally.

        Distinct-point quartic correlations needed by the momentum-space
        interaction can therefore be contracted with the same finite
        auxiliary space as the norm.  Coincident normal pairs should use the
        single ``"density"`` insertion to avoid an artificial contact term.
        """
        entries = list(insertions)
        (
            energies,
            _radial_widths,
            shell_generators,
            _right_before,
            _left_after,
            norm,
        ) = self._radial_shell_data()
        if abs(norm) <= np.finfo(float).tiny:
            raise FloatingPointError("the hLETTA norm is numerically zero.")

        grouped = [[] for _ in range(len(energies))]
        valid_operators = {"annihilation", "creation", "density"}
        for entry in entries:
            if len(entry) != 3:
                raise ValueError(
                    "each insertion must be (radial_index, theta, operator)."
                )
            radial_index, theta, operator = entry
            radial_index = int(radial_index)
            theta = float(theta)
            operator = str(operator).lower()
            if not (0 <= radial_index < len(energies)):
                raise IndexError("radial_index is outside the quadrature.")
            if not (0.0 <= theta <= 2.0 * np.pi):
                raise ValueError("theta must lie in [0, 2*pi].")
            if operator not in valid_operators:
                raise ValueError(
                    "operator must be annihilation, creation, or density."
                )
            grouped[radial_index].append((theta, operator))

        left, right = self.boundary_vectors()
        environment = right
        identity = np.eye(self.effective_bond_dim, dtype=np.complex128)
        for radial_index, generator in enumerate(shell_generators):
            previous_angle = 0.0
            for theta, operator in sorted(grouped[radial_index]):
                environment = (
                    expm((theta - previous_angle) * generator) @ environment
                )
                physical = self.physical_insertion(
                    float(energies[radial_index]), theta
                )
                if operator == "annihilation":
                    field_operator = np.kron(physical, identity)
                elif operator == "creation":
                    field_operator = np.kron(identity, physical.conj())
                else:
                    field_operator = np.kron(physical, physical.conj())
                environment = field_operator @ environment
                previous_angle = theta
            environment = (
                expm((2.0 * np.pi - previous_angle) * generator)
                @ environment
            )
        return complex(np.vdot(left, environment)) / norm

    def antipodal_pair_expectation(self, weight=1.0, *, angular_points=None):
        r"""Return the ordered antipodal anomalous integral.

        The convention is

        $$
        \int_0^{E_c}dE\int_0^\pi d\theta\,w(E)
        \langle\psi(E,\theta)\psi(E,\theta+\pi)\rangle.
        $$

        Only one representative of each antipodal pair is included.  The
        phase-only angular dependence of this minimal ansatz makes the shell
        transfer generator angle independent, while the two ket insertions
        retain their full ordered propagation.
        """
        if callable(weight):
            weight_function = weight
        else:
            constant = complex(weight)

            def weight_function(_energy):
                return constant

        if angular_points is None:
            angular_points = self.contraction.angular_points
        angular_points = int(angular_points)
        if angular_points < 2:
            raise ValueError("angular_points must be at least two.")
        nodes, weights = leggauss(angular_points)
        angles = 0.5 * np.pi * (nodes + 1.0)
        angle_widths = 0.5 * np.pi * weights

        (
            energies,
            radial_widths,
            shell_generators,
            right_before,
            left_after,
            norm,
        ) = self._radial_shell_data()
        if abs(norm) <= np.finfo(float).tiny:
            raise FloatingPointError("the hLETTA norm is numerically zero.")

        identity = np.eye(self.effective_bond_dim, dtype=np.complex128)
        total = 0.0j
        for (
            energy,
            radial_width,
            generator,
            environment,
            covector,
        ) in zip(
            energies,
            radial_widths,
            shell_generators,
            right_before,
            left_after,
        ):
            middle = expm(np.pi * generator)
            for angle, angle_width in zip(angles, angle_widths):
                first = np.kron(
                    self.physical_insertion(float(energy), float(angle)),
                    identity,
                )
                second_angle = float(angle) + np.pi
                second = np.kron(
                    self.physical_insertion(float(energy), second_angle),
                    identity,
                )
                inserted = (
                    expm((2.0 * np.pi - second_angle) * generator)
                    @ second
                    @ middle
                    @ first
                    @ expm(float(angle) * generator)
                    @ environment
                )
                total += (
                    float(radial_width)
                    * float(angle_width)
                    * complex(weight_function(float(energy)))
                    * np.vdot(covector, inserted)
                )
        return total / norm

    def bogoliubov_shell_functional(self, model):
        r"""Evaluate the smooth-potential quadratic shell functional.

        The shell coordinate is the free energy ``E = kinetic_prefactor*k**2``
        and must use a cutoff in the same units.  With

        ``A(E) = E + density * V(k)`` and ``B(E) = density * V(k)``, this
        returns

        $$
        \mathcal E_{MF}+\int dE\,d\theta\,A(E)\langle n\rangle
        -2\operatorname{Re}\int dE\int_0^\pi d\theta\,
        B(E)\langle\psi_\theta\psi_{\theta+\pi}\rangle.
        $$

        This is an energy functional in the normalized shell-field convention,
        not yet a thermodynamic energy density.  Matching it quantitatively to
        the real-space energy per area requires an explicit finite-area
        regulator followed by the thermodynamic limit.  It must therefore not
        be compared variationally with ``model.bogoliubov_energy_density``.
        """
        if not isinstance(model, GaussianPotentialBoseGas2D):
            raise TypeError("model must be GaussianPotentialBoseGas2D.")

        def interaction(energy):
            momentum = np.sqrt(float(energy) / model.kinetic_prefactor)
            return model.density * model.interaction_momentum(momentum)

        normal = self.additive_one_body_expectation(
            lambda energy: energy + interaction(energy)
        )
        anomalous = self.antipodal_pair_expectation(interaction)
        value = model.mean_field_energy_density + normal - 2.0 * anomalous.real
        return float(np.real_if_close(value).real)


@dataclass
class D2M1NestedCLETTA2D:
    r"""Genuinely nested ``D=2, M=1`` radial/angular cLETTA contraction.

    For a radial cell of quadrature width ``dE``, the inner angular cLETTA uses

    $$
    Q_c=dE\,I_m\otimes Q-\kappa N_m\otimes I_D,
    $$

    $$
    R_c(\theta)=e^{i\ell\theta}\left[
    \sqrt{dE}\,I_m\otimes R+\sqrt\kappa\,a_m\otimes I_D
    +\sqrt{dE}\,a_m^\dagger\otimes S\right].
    $$

    The angular memory therefore propagates over a finite angle at order one,
    while its coupling to a measure-zero energy shell is ``sqrt(dE)``.  After
    the exact angular double-layer contraction, the memory is projected back
    to its vacuum.  This gives a ``D**2 x D**2`` shell channel

    $$
    \Phi_E(dE)=P_{m=0}\exp(2\pi\mathbb T_E)P_{m=0}^\dagger,
    $$

    and the outer contraction is the energy-ordered product of these channels.
    Consequently ``Phi_E(dE) = I + O(dE)`` and the radial continuum limit is
    well defined without weakening the inner angular memory decay.

    ``M=1`` denotes one angular memory channel truncated to occupations zero
    and one.  Cross-energy ties ``K(E,E')`` require an additional outer memory
    channel and are deliberately not folded into this minimal benchmark.
    """

    contraction: HierarchicalShellContraction
    q_matrix: np.ndarray
    r_matrix: np.ndarray
    tie_matrix: np.ndarray
    angular_memory_decay: float
    radial_decay: float = 0.0
    angular_momentum: int = 0
    field_phase: float = 0.0
    replication_scale: float = 1.0
    generator_step: float = 1.0e-3

    def __post_init__(self):
        self.q_matrix = np.asarray(self.q_matrix, dtype=np.complex128)
        self.r_matrix = np.asarray(self.r_matrix, dtype=np.complex128)
        self.tie_matrix = np.asarray(self.tie_matrix, dtype=np.complex128)
        for name, matrix in (
            ("q_matrix", self.q_matrix),
            ("r_matrix", self.r_matrix),
            ("tie_matrix", self.tie_matrix),
        ):
            if matrix.shape != (2, 2):
                raise ValueError(f"{name} must have shape (2, 2) for D=2.")
        self.angular_memory_decay = float(self.angular_memory_decay)
        self.radial_decay = float(self.radial_decay)
        self.angular_momentum = int(self.angular_momentum)
        self.field_phase = float(self.field_phase)
        self.replication_scale = float(self.replication_scale)
        self.generator_step = float(self.generator_step)
        if (
            not np.isfinite(self.angular_memory_decay)
            or self.angular_memory_decay <= 0.0
        ):
            raise ValueError(
                "angular_memory_decay must be finite and positive."
            )
        if not np.isfinite(self.radial_decay) or self.radial_decay < 0.0:
            raise ValueError("radial_decay must be finite and non-negative.")
        if not np.isfinite(self.field_phase):
            raise ValueError("field_phase must be finite.")
        if not np.isfinite(self.replication_scale) or self.replication_scale <= 0.0:
            raise ValueError("replication_scale must be finite and positive.")
        if not np.isfinite(self.generator_step) or self.generator_step <= 0.0:
            raise ValueError("generator_step must be finite and positive.")

    @property
    def bond_dim(self):
        return 2

    @property
    def num_angular_memory_modes(self):
        return 1

    @property
    def memory_dim(self):
        return 2

    @property
    def inner_bond_dim(self):
        return self.bond_dim * self.memory_dim

    @property
    def inner_transfer_dim(self):
        return self.inner_bond_dim**2

    @property
    def outer_transfer_dim(self):
        return self.bond_dim**2

    def radial_envelope(self, energy):
        scaled_energy = float(energy) / self.contraction.energy_cutoff
        return float(np.exp(-self.radial_decay * scaled_energy))

    def _combined_shell_matrices_from_amplitude(
        self, energy, shell_amplitude, *, theta=0.0
    ):
        shell_amplitude = float(shell_amplitude)
        weak_scale = shell_amplitude * self.radial_envelope(energy)
        q_combined, r_combined = cletta_memory_matrices(
            shell_amplitude**2 * self.q_matrix,
            weak_scale * self.r_matrix,
            weak_scale * self.tie_matrix,
            self.angular_memory_decay,
            memory_dim=self.memory_dim,
        )
        phase = np.exp(
            1.0j
            * (
                self.angular_momentum * float(theta)
                + self.field_phase
            )
        )
        return q_combined, phase * r_combined

    def combined_shell_matrices(self, energy, radial_width, *, theta=0.0):
        """Return the weak-shell ``4 x 4`` inner matrices ``(Q_c, R_c)``."""
        radial_width = float(radial_width)
        if radial_width <= 0.0:
            raise ValueError("radial_width must be positive.")
        return self._combined_shell_matrices_from_amplitude(
            energy, np.sqrt(radial_width), theta=theta
        )

    def _memory_vacuum_embedding(self):
        """Embed the outer double layer into the inner memory vacuum."""
        embedding = np.zeros(
            (self.inner_transfer_dim, self.outer_transfer_dim),
            dtype=np.complex128,
        )
        for ket_outer in range(self.bond_dim):
            for bra_outer in range(self.bond_dim):
                ket_combined = ket_outer
                bra_combined = bra_outer
                inner_index = (
                    ket_combined * self.inner_bond_dim + bra_combined
                )
                outer_index = ket_outer * self.bond_dim + bra_outer
                embedding[inner_index, outer_index] = 1.0
        return embedding

    def _shell_channel_from_amplitude(
        self,
        energy,
        shell_amplitude,
        *,
        counting_source=0.0,
        counting_weight=1.0,
    ):
        transfer, _ = self._inner_transfer_from_amplitude(
            energy,
            shell_amplitude,
            counting_source=counting_source,
            counting_weight=counting_weight,
        )
        embedding = self._memory_vacuum_embedding()
        return embedding.conj().T @ expm(2.0 * np.pi * transfer) @ embedding

    def _inner_transfer_from_amplitude(
        self,
        energy,
        shell_amplitude,
        *,
        counting_source=0.0,
        counting_weight=1.0,
    ):
        q_combined, r_combined = self._combined_shell_matrices_from_amplitude(
            energy, shell_amplitude
        )
        identity = np.eye(self.inner_bond_dim, dtype=np.complex128)
        jump = np.kron(r_combined, r_combined.conj())
        transfer = (
            np.kron(q_combined, identity)
            + np.kron(identity, q_combined.conj())
            + np.exp(
                complex(counting_source) * complex(counting_weight)
            )
            * jump
        )
        return transfer, jump

    def _shell_channel_source_derivative_from_amplitude(
        self, energy, shell_amplitude, *, weight=1.0
    ):
        transfer, jump = self._inner_transfer_from_amplitude(
            energy, shell_amplitude
        )
        derivative = complex(weight) * jump
        channel_derivative = expm_frechet(
            2.0 * np.pi * transfer,
            2.0 * np.pi * derivative,
            compute_expm=False,
        )
        embedding = self._memory_vacuum_embedding()
        return embedding.conj().T @ channel_derivative @ embedding

    def _inner_insertion_channel_from_amplitude(
        self, energy, theta, operator, shell_amplitude
    ):
        transfer, _ = self._inner_transfer_from_amplitude(
            energy, shell_amplitude
        )
        _, physical = self._combined_shell_matrices_from_amplitude(
            energy, shell_amplitude, theta=theta
        )
        identity = np.eye(self.inner_bond_dim, dtype=np.complex128)
        operator = str(operator).lower()
        if operator == "annihilation":
            insertion = np.kron(physical, identity)
        elif operator == "creation":
            insertion = np.kron(identity, physical.conj())
        elif operator == "density":
            insertion = np.kron(physical, physical.conj())
        else:
            raise ValueError(
                "operator must be annihilation, creation, or density."
            )
        embedding = self._memory_vacuum_embedding()
        channel = (
            expm((2.0 * np.pi - float(theta)) * transfer)
            @ insertion
            @ expm(float(theta) * transfer)
        )
        return embedding.conj().T @ channel @ embedding

    def effective_field_insertion(self, energy, theta, operator):
        r"""Return a continuum outer map for one physical field insertion.

        Creation and annihilation maps are coefficients of ``sqrt(dE)``;
        the coincident normal-density map is the coefficient of ``dE``.
        """
        energy = float(energy)
        theta = float(theta) % (2.0 * np.pi)
        if not (0.0 <= energy <= self.contraction.energy_cutoff):
            raise ValueError("energy lies outside the hLETTA cutoff.")
        operator = str(operator).lower()
        step = self.generator_step
        plus = self._inner_insertion_channel_from_amplitude(
            energy, theta, operator, step
        )
        minus = self._inner_insertion_channel_from_amplitude(
            energy, theta, operator, -step
        )
        if operator in {"annihilation", "creation"}:
            insertion = (plus - minus) / (2.0 * step)
            return np.sqrt(self.replication_scale) * insertion
        if operator == "density":
            zero = self._inner_insertion_channel_from_amplitude(
                energy, theta, operator, 0.0
            )
            insertion = (
                plus + minus - 2.0 * zero
            ) / (2.0 * step**2)
            return self.replication_scale * insertion
        raise ValueError("operator must be annihilation, creation, or density.")

    def _inner_two_field_channel_from_amplitude(
        self,
        energy,
        first_theta,
        first_operator,
        second_theta,
        second_operator,
        shell_amplitude,
    ):
        """Return one inner-shell channel with two angular field insertions."""
        entries = sorted(
            [
                (float(first_theta), str(first_operator).lower()),
                (float(second_theta), str(second_operator).lower()),
            ]
        )
        if not (0.0 <= entries[0][0] <= entries[1][0] <= 2.0 * np.pi):
            raise ValueError("field angles must lie in [0, 2*pi].")
        transfer, _ = self._inner_transfer_from_amplitude(
            energy, shell_amplitude
        )
        identity = np.eye(self.inner_bond_dim, dtype=np.complex128)

        def insertion(theta, operator):
            _, physical = self._combined_shell_matrices_from_amplitude(
                energy, shell_amplitude, theta=theta
            )
            if operator == "annihilation":
                return np.kron(physical, identity)
            if operator == "creation":
                return np.kron(identity, physical.conj())
            raise ValueError("pair operators must be annihilation or creation.")

        first_angle, first_kind = entries[0]
        second_angle, second_kind = entries[1]
        first = insertion(first_angle, first_kind)
        second = insertion(second_angle, second_kind)
        embedding = self._memory_vacuum_embedding()
        channel = (
            expm((2.0 * np.pi - second_angle) * transfer)
            @ second
            @ expm((second_angle - first_angle) * transfer)
            @ first
            @ expm(first_angle * transfer)
        )
        return embedding.conj().T @ channel @ embedding

    def effective_pair_insertion(
        self,
        energy,
        first_theta,
        first_operator,
        second_theta,
        second_operator,
    ):
        r"""Return the ``dE`` coefficient of a same-shell field pair.

        Both physical fields scale as ``sqrt(dE)``.  A centered second
        derivative in that amplitude extracts their joint continuum map while
        retaining the exact angular-memory propagation between them.
        """
        step = self.generator_step

        def channel(amplitude):
            return self._inner_two_field_channel_from_amplitude(
                float(energy),
                first_theta,
                first_operator,
                second_theta,
                second_operator,
                amplitude,
            )

        insertion = (
            channel(step) + channel(-step) - 2.0 * channel(0.0)
        ) / (2.0 * step**2)
        return self.replication_scale * insertion

    def outer_propagator(self, energy_start, energy_stop, *, points=6):
        r"""Return ``P_E exp integral_(start)^(stop) dE K_E``."""
        energy_start = float(energy_start)
        energy_stop = float(energy_stop)
        points = int(points)
        if not (
            0.0
            <= energy_start
            <= energy_stop
            <= self.contraction.energy_cutoff
        ):
            raise ValueError("invalid ordered energy interval.")
        if points < 2:
            raise ValueError("points must be at least two.")
        result = np.eye(self.outer_transfer_dim, dtype=np.complex128)
        if energy_stop == energy_start:
            return result
        nodes, weights = leggauss(points)
        half_width = 0.5 * (energy_stop - energy_start)
        midpoint = 0.5 * (energy_stop + energy_start)
        energies = midpoint + half_width * nodes
        widths = half_width * weights
        for energy, width in zip(energies, widths):
            result = (
                expm(
                    float(width)
                    * self.replication_scale
                    * self.effective_outer_generator(energy)
                )
                @ result
            )
        return result

    def field_correlation(self, insertions, *, propagation_points=6):
        r"""Contract fields at arbitrary distinct momentum-shell coordinates.

        Each entry is ``(energy, theta, operator)``.  Equal-energy multi-field
        insertions require a joint inner-shell channel and are rejected here;
        they form a measure-zero set in the density-transfer integrals.
        """
        entries = []
        for entry in insertions:
            if len(entry) != 3:
                raise ValueError(
                    "each insertion must be (energy, theta, operator)."
                )
            energy, theta, operator = entry
            entries.append((float(energy), float(theta), str(operator).lower()))
        entries.sort(key=lambda item: (item[0], item[1]))
        for first, second in zip(entries, entries[1:]):
            if np.isclose(first[0], second[0], rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    "equal-energy fields need a joint inner-shell insertion."
                )

        left, environment = self.boundary_vectors()
        previous_energy = 0.0
        for energy, theta, operator in entries:
            environment = (
                self.outer_propagator(
                    previous_energy,
                    energy,
                    points=propagation_points,
                )
                @ environment
            )
            environment = (
                self.effective_field_insertion(energy, theta, operator)
                @ environment
            )
            previous_energy = energy
        environment = (
            self.outer_propagator(
                previous_energy,
                self.contraction.energy_cutoff,
                points=propagation_points,
            )
            @ environment
        )
        norm = self.norm()
        if abs(norm) <= np.finfo(float).tiny:
            raise FloatingPointError("the nested hLETTA norm is numerically zero.")
        return complex(np.vdot(left, environment)) / norm

    def local_density(self, energy, theta=0.0, *, propagation_points=6):
        """Return ``<psi^dagger(E,theta) psi(E,theta)>``."""
        left, right = self.boundary_vectors()
        before = self.outer_propagator(
            0.0, float(energy), points=propagation_points
        )
        after = self.outer_propagator(
            float(energy),
            self.contraction.energy_cutoff,
            points=propagation_points,
        )
        insertion = self.effective_field_insertion(energy, theta, "density")
        value = np.vdot(left, after @ insertion @ before @ right)
        norm = self.norm()
        return complex(value) / norm

    def antipodal_pair_expectation(
        self,
        weight=1.0,
        *,
        angular_points=None,
        propagation_points=3,
    ):
        r"""Contract ``integral dE integral_0^pi dtheta w(E) <b_k b_-k>``.

        This uses a joint inner-shell insertion because antipodal momenta have
        the same energy and therefore cannot be contracted as two independent
        outer-energy insertions.
        """
        if callable(weight):
            weight_function = weight
        else:
            constant = complex(weight)

            def weight_function(_energy):
                return constant

        if angular_points is None:
            angular_points = self.contraction.angular_points
        angular_points = int(angular_points)
        propagation_points = int(propagation_points)
        if angular_points < 2:
            raise ValueError("angular_points must be at least two.")
        if propagation_points < 2:
            raise ValueError("propagation_points must be at least two.")
        angular_nodes, angular_weights = leggauss(angular_points)
        angles = 0.5 * np.pi * (angular_nodes + 1.0)
        angle_widths = 0.5 * np.pi * angular_weights
        energies, energy_widths = self.contraction.radial_quadrature()
        left, right = self.boundary_vectors()
        norm = self.norm()
        if abs(norm) <= np.finfo(float).tiny:
            raise FloatingPointError("the nested hLETTA norm is numerically zero.")

        total = 0.0j
        for energy, energy_width in zip(energies, energy_widths):
            before = self.outer_propagator(
                0.0, float(energy), points=propagation_points
            )
            after = self.outer_propagator(
                float(energy),
                self.contraction.energy_cutoff,
                points=propagation_points,
            )
            for theta, theta_width in zip(angles, angle_widths):
                pair = self.effective_pair_insertion(
                    float(energy),
                    float(theta),
                    "annihilation",
                    float(theta) + np.pi,
                    "annihilation",
                )
                total += (
                    float(energy_width)
                    * float(theta_width)
                    * complex(weight_function(float(energy)))
                    * np.vdot(left, after @ pair @ before @ right)
                    / norm
                )
        return total

    @staticmethod
    def _polar_coordinates(vector):
        momentum = float(np.hypot(vector[0], vector[1]))
        theta = float(np.arctan2(vector[1], vector[0]) % (2.0 * np.pi))
        return momentum, theta

    def normal_ordered_density_structure(
        self,
        momentum_transfer,
        *,
        kinetic_prefactor=1.0,
        area=1.0,
        radial_points=2,
        angular_points=3,
        propagation_points=3,
    ):
        r"""Contract ``<:rho_q rho_-q:>/area`` for ``q`` along the x axis.

        The state is isotropic for the default angular parameterization, so a
        fixed transfer direction represents the continuous directional
        average.  The two density modes are expanded as

        $$
        :\rho_{\mathbf q}\rho_{-\mathbf q}:
        =\int_{\mathbf k,\mathbf p}
        a^\dagger_{\mathbf{k+q}}a^\dagger_{\mathbf{p-q}}
        a_{\mathbf k}a_{\mathbf p}.
        $$

        Radial and angular nodes are continuum integration points.  Different
        Gauss rules are used for ``k`` and ``p`` so coincident-energy shells,
        which have zero measure in the integral, are not assigned artificial
        weight.
        """
        momentum_transfer = float(momentum_transfer)
        kinetic_prefactor = float(kinetic_prefactor)
        area = float(area)
        radial_points = int(radial_points)
        angular_points = int(angular_points)
        propagation_points = int(propagation_points)
        if momentum_transfer < 0.0:
            raise ValueError("momentum_transfer must be non-negative.")
        if kinetic_prefactor <= 0.0 or area <= 0.0:
            raise ValueError("kinetic_prefactor and area must be positive.")
        if radial_points < 2 or angular_points < 3:
            raise ValueError(
                "radial_points must be at least two and angular_points at least three."
            )
        if propagation_points < 2:
            raise ValueError("propagation_points must be at least two.")

        cutoff = self.contraction.energy_cutoff
        momentum_cutoff = np.sqrt(cutoff / kinetic_prefactor)

        def momentum_points(num_radial):
            radial_nodes, radial_weights = leggauss(num_radial)
            energies = 0.5 * cutoff * (radial_nodes + 1.0)
            energy_weights = 0.5 * cutoff * radial_weights
            angular_nodes, angular_weights = leggauss(angular_points)
            angles = np.pi * (angular_nodes + 1.0)
            angle_weights = np.pi * angular_weights
            points = []
            for energy, energy_weight in zip(energies, energy_weights):
                momentum = np.sqrt(float(energy) / kinetic_prefactor)
                for theta, angle_weight in zip(angles, angle_weights):
                    vector = np.array(
                        [momentum * np.cos(theta), momentum * np.sin(theta)]
                    )
                    points.append(
                        (vector, float(energy_weight * angle_weight))
                    )
            return points

        first_points = momentum_points(radial_points)
        second_points = momentum_points(radial_points + 1)
        transfer_vector = np.array([momentum_transfer, 0.0])

        samples = []
        energy_values = {0.0, float(cutoff)}
        for first_vector, first_weight in first_points:
            first_shifted = first_vector + transfer_vector
            first_shifted_norm, first_shifted_theta = self._polar_coordinates(
                first_shifted
            )
            if first_shifted_norm >= momentum_cutoff:
                continue
            first_energy = kinetic_prefactor * float(first_vector @ first_vector)
            first_theta = self._polar_coordinates(first_vector)[1]
            first_shifted_energy = kinetic_prefactor * first_shifted_norm**2
            for second_vector, second_weight in second_points:
                second_shifted = second_vector - transfer_vector
                second_shifted_norm, second_shifted_theta = self._polar_coordinates(
                    second_shifted
                )
                if second_shifted_norm >= momentum_cutoff:
                    continue
                second_energy = kinetic_prefactor * float(
                    second_vector @ second_vector
                )
                second_theta = self._polar_coordinates(second_vector)[1]
                second_shifted_energy = (
                    kinetic_prefactor * second_shifted_norm**2
                )
                coordinates = [
                    (first_shifted_energy, first_shifted_theta, "creation"),
                    (second_shifted_energy, second_shifted_theta, "creation"),
                    (first_energy, first_theta, "annihilation"),
                    (second_energy, second_theta, "annihilation"),
                ]
                ordered_energies = sorted(item[0] for item in coordinates)
                if any(
                    np.isclose(a_value, b_value, rtol=0.0, atol=1.0e-11)
                    for a_value, b_value in zip(
                        ordered_energies, ordered_energies[1:]
                    )
                ):
                    continue
                samples.append(
                    (coordinates, float(first_weight * second_weight))
                )
                energy_values.update(float(item[0]) for item in coordinates)

        rounded_to_energy = {}
        for energy in energy_values:
            rounded_to_energy.setdefault(round(float(energy), 13), float(energy))
        energy_axis = np.array(sorted(rounded_to_energy.values()))
        energy_indices = {
            round(float(energy), 13): index
            for index, energy in enumerate(energy_axis)
        }
        segment_maps = [
            self.outer_propagator(
                energy_axis[index],
                energy_axis[index + 1],
                points=propagation_points,
            )
            for index in range(len(energy_axis) - 1)
        ]
        propagator_cache = {}

        def propagator(first_energy, second_energy):
            first_index = energy_indices[round(float(first_energy), 13)]
            second_index = energy_indices[round(float(second_energy), 13)]
            key = (first_index, second_index)
            if key not in propagator_cache:
                result = np.eye(
                    self.outer_transfer_dim, dtype=np.complex128
                )
                for index in range(first_index, second_index):
                    result = segment_maps[index] @ result
                propagator_cache[key] = result
            return propagator_cache[key]

        insertion_cache = {}

        def insertion_map(energy, theta, operator):
            key = (round(float(energy), 13), round(float(theta), 13), operator)
            if key not in insertion_cache:
                insertion_cache[key] = self.effective_field_insertion(
                    energy, theta, operator
                )
            return insertion_cache[key]

        left, right = self.boundary_vectors()
        norm = self.norm()
        total = 0.0j
        for coordinates, sample_weight in samples:
            environment = right
            previous_energy = 0.0
            for energy, theta, operator in sorted(
                coordinates, key=lambda item: (item[0], item[1])
            ):
                environment = (
                    propagator(previous_energy, energy) @ environment
                )
                environment = (
                    insertion_map(energy, theta, operator) @ environment
                )
                previous_energy = energy
            environment = propagator(previous_energy, cutoff) @ environment
            total += sample_weight * np.vdot(left, environment) / norm
        result = total / (area * self.replication_scale)
        if abs(result.imag) > 2.0e-6 * max(1.0, abs(result.real)):
            raise FloatingPointError(
                "density structure acquired an imaginary part."
            )
        return float(result.real)

    def shell_diagonal_density_structure(
        self,
        momentum_transfer,
        *,
        kinetic_prefactor=1.0,
        area=1.0,
        radial_points=2,
        angular_points=3,
        propagation_points=3,
    ):
        r"""Contract the shell-diagonal part of ``<:rho_q rho_-q:>/A``.

        The inner angular memory can emit an antipodal pair at one exact
        energy shell.  Its two-point kernel therefore contains radial delta
        support that a quadrature over four distinct field coordinates does
        not see.  This method contracts the resulting pair--pair channel

        $$
        \int_k\langle
        b^\dagger_{k+q}b^\dagger_{-k-q}b_kb_{-k}\rangle
        $$

        together with the shell-diagonal normal exchange channel.  These are
        continuum contact contributions, not extra quadrature orbitals.
        """
        momentum_transfer = float(momentum_transfer)
        kinetic_prefactor = float(kinetic_prefactor)
        area = float(area)
        radial_points = int(radial_points)
        angular_points = int(angular_points)
        propagation_points = int(propagation_points)
        if momentum_transfer < 0.0:
            raise ValueError("momentum_transfer must be non-negative.")
        if kinetic_prefactor <= 0.0 or area <= 0.0:
            raise ValueError("kinetic_prefactor and area must be positive.")
        if radial_points < 2 or angular_points < 3:
            raise ValueError(
                "radial_points must be at least two and angular_points at least three."
            )
        if propagation_points < 2:
            raise ValueError("propagation_points must be at least two.")

        cutoff = self.contraction.energy_cutoff
        momentum_cutoff = np.sqrt(cutoff / kinetic_prefactor)
        radial_nodes, radial_weights = leggauss(radial_points)
        energies = 0.5 * cutoff * (radial_nodes + 1.0)
        energy_weights = 0.5 * cutoff * radial_weights
        angular_nodes, angular_weights = leggauss(angular_points)
        angles = np.pi * (angular_nodes + 1.0)
        angle_weights = np.pi * angular_weights
        transfer_vector = np.array([momentum_transfer, 0.0])
        left, right = self.boundary_vectors()
        norm = self.norm()
        propagator_cache = {}

        def propagator(first, second):
            key = (round(float(first), 13), round(float(second), 13))
            if key not in propagator_cache:
                propagator_cache[key] = self.outer_propagator(
                    first, second, points=propagation_points
                )
            return propagator_cache[key]

        insertion_cache = {}

        def density_map(energy, theta):
            key = ("density", round(float(energy), 13), round(float(theta), 13))
            if key not in insertion_cache:
                insertion_cache[key] = self.effective_field_insertion(
                    energy, theta, "density"
                )
            return insertion_cache[key]

        def pair_map(energy, theta, operator):
            representative = float(theta) % np.pi
            key = (
                operator,
                round(float(energy), 13),
                round(representative, 13),
            )
            if key not in insertion_cache:
                insertion_cache[key] = self.effective_pair_insertion(
                    energy,
                    representative,
                    operator,
                    representative + np.pi,
                    operator,
                )
            return insertion_cache[key]

        def ordered_two_map_correlation(
            first_energy, first_map, second_energy, second_map
        ):
            if np.isclose(
                first_energy, second_energy, rtol=0.0, atol=1.0e-11
            ):
                return 0.0j
            if first_energy > second_energy:
                first_energy, second_energy = second_energy, first_energy
                first_map, second_map = second_map, first_map
            environment = propagator(0.0, first_energy) @ right
            environment = first_map @ environment
            environment = (
                propagator(first_energy, second_energy) @ environment
            )
            environment = second_map @ environment
            environment = propagator(second_energy, cutoff) @ environment
            return complex(np.vdot(left, environment)) / norm

        total = 0.0j
        for energy, energy_weight in zip(energies, energy_weights):
            momentum = np.sqrt(float(energy) / kinetic_prefactor)
            for theta, theta_weight in zip(angles, angle_weights):
                vector = np.array(
                    [momentum * np.cos(theta), momentum * np.sin(theta)]
                )
                shifted = vector + transfer_vector
                shifted_norm, shifted_theta = self._polar_coordinates(shifted)
                if shifted_norm >= momentum_cutoff:
                    continue
                shifted_energy = kinetic_prefactor * shifted_norm**2
                normal = ordered_two_map_correlation(
                    float(energy),
                    density_map(float(energy), float(theta)),
                    shifted_energy,
                    density_map(shifted_energy, shifted_theta),
                )
                pairing = ordered_two_map_correlation(
                    float(energy),
                    pair_map(float(energy), float(theta), "annihilation"),
                    shifted_energy,
                    pair_map(shifted_energy, shifted_theta, "creation"),
                )
                total += (
                    float(energy_weight)
                    * float(theta_weight)
                    * (normal + pairing)
                )
        result = total / (area * self.replication_scale)
        if abs(result.imag) > 2.0e-6 * max(1.0, abs(result.real)):
            raise FloatingPointError(
                "shell-diagonal density structure acquired an imaginary part."
            )
        return float(result.real)

    def evaluate_rank_one_density_transfer(
        self,
        channel,
        *,
        kinetic_prefactor=1.0,
        area=1.0,
        structure_radial_points=2,
        structure_angular_points=3,
        propagation_points=3,
    ):
        r"""Evaluate the projected interacting hLETTA energy and return self.

        The Hamiltonian is the cutoff theory

        $$
        H=\int dE\,d\theta\,E\,\psi^\dagger\psi
        +\frac12\int_{\mathbf q}|u(q)|^2
        :\rho_{\mathbf q}\rho_{-\mathbf q}: .
        $$

        The method populates ``density_transfer_momenta``,
        ``density_transfer_structure``, ``particle_density``,
        ``kinetic_energy_density``, ``interaction_energy_density``, and
        ``energy_density``.  No condensate c-number is added: fixed density
        must be imposed by varying the hLETTA parameters themselves.
        """
        if not isinstance(channel, RankOneDensityTransferChannel2D):
            raise TypeError("channel must be RankOneDensityTransferChannel2D.")
        area = float(area)
        if area <= 0.0:
            raise ValueError("area must be positive.")
        momenta, _ = channel.radial_quadrature()
        structure = np.asarray(
            [
                self.normal_ordered_density_structure(
                    momentum_transfer,
                    kinetic_prefactor=kinetic_prefactor,
                    area=area,
                    radial_points=structure_radial_points,
                    angular_points=structure_angular_points,
                    propagation_points=propagation_points,
                )
                for momentum_transfer in momenta
            ]
        )
        self.density_transfer_momenta = momenta
        self.density_transfer_structure = structure
        self.particle_density = self.particle_number() / area
        self.kinetic_energy_density = self.kinetic_energy() / area
        self.interaction_energy_density = (
            channel.normal_ordered_interaction_energy_density(structure)
        )
        self.energy_density = (
            self.kinetic_energy_density + self.interaction_energy_density
        )
        return self

    def evaluate_condensate_shifted_rank_one(
        self,
        channel,
        *,
        condensate_density,
        kinetic_prefactor=1.0,
        area=1.0,
        structure_radial_points=2,
        structure_angular_points=3,
        pair_angular_points=3,
        propagation_points=3,
        include_smooth_structure=True,
    ):
        r"""Evaluate a condensate plus parity-even hLETTA fluctuation state.

        With ``a_0 -> sqrt(A n0)`` and nonzero-momentum fluctuation operators
        ``b_k``, the interaction energy density retained here is

        $$
        \frac{V(0)n_0^2}{2}
        +\frac{n_0}{A}\int_k [V(0)+V(k)]\langle b_k^\dagger b_k\rangle
        +\frac{n_0}{A}\operatorname{Re}\int_k
          V(k)\langle b_k b_{-k}\rangle
        +\mathcal E_4.
        $$

        ``E4`` is the fully contracted normal-ordered four-fluctuation term.
        Odd fluctuation terms vanish for the parity-even parameterization
        produced by :func:`fixed_density_nested_hletta_state`.
        """
        if not isinstance(channel, RankOneDensityTransferChannel2D):
            raise TypeError("channel must be RankOneDensityTransferChannel2D.")
        if not getattr(self, "parity_even", False):
            raise ValueError(
                "condensate shifting requires a parity-even fluctuation ansatz."
            )
        condensate_density = float(condensate_density)
        kinetic_prefactor = float(kinetic_prefactor)
        area = float(area)
        if condensate_density < 0.0 or area <= 0.0:
            raise ValueError("condensate_density must be non-negative and area positive.")
        if kinetic_prefactor <= 0.0:
            raise ValueError("kinetic_prefactor must be positive.")

        momenta, _ = channel.radial_quadrature()
        if include_smooth_structure:
            smooth_structure = np.asarray(
                [
                    self.normal_ordered_density_structure(
                        momentum_transfer,
                        kinetic_prefactor=kinetic_prefactor,
                        area=area,
                        radial_points=structure_radial_points,
                        angular_points=structure_angular_points,
                        propagation_points=propagation_points,
                    )
                    for momentum_transfer in momenta
                ]
            )
        else:
            smooth_structure = np.zeros_like(momenta)
        shell_diagonal_structure = np.asarray(
            [
                self.shell_diagonal_density_structure(
                    momentum_transfer,
                    kinetic_prefactor=kinetic_prefactor,
                    area=area,
                    radial_points=structure_radial_points,
                    angular_points=structure_angular_points,
                    propagation_points=propagation_points,
                )
                for momentum_transfer in momenta
            ]
        )
        fluctuation_structure = smooth_structure + shell_diagonal_structure
        profile_zero = complex(
            np.asarray(channel.radial_profile(0.0), dtype=np.complex128).item()
        )
        potential_zero = abs(profile_zero) ** 2

        def potential_at_energy(energy):
            momentum = np.sqrt(float(energy) / kinetic_prefactor)
            profile = complex(
                np.asarray(
                    channel.radial_profile(momentum), dtype=np.complex128
                ).item()
            )
            return abs(profile) ** 2

        fluctuation_number = self.particle_number()
        normal_shift = condensate_density * float(
            np.real_if_close(
                self.additive_one_body_expectation(
                    lambda energy: potential_zero + potential_at_energy(energy)
                )
            ).real
        ) / area
        half_plane_pair = self.antipodal_pair_expectation(
            potential_at_energy,
            angular_points=pair_angular_points,
            propagation_points=propagation_points,
        )
        anomalous_shift = (
            2.0 * condensate_density * float(half_plane_pair.real) / area
        )
        condensate_mean_field = (
            0.5 * potential_zero * condensate_density**2
        )
        fluctuation_quartic = (
            channel.normal_ordered_interaction_energy_density(
                fluctuation_structure
            )
        )

        self.density_transfer_momenta = momenta
        self.density_transfer_structure = fluctuation_structure
        self.smooth_density_transfer_structure = smooth_structure
        self.shell_diagonal_density_transfer_structure = (
            shell_diagonal_structure
        )
        self.includes_smooth_density_structure = bool(
            include_smooth_structure
        )
        self.condensate_density = condensate_density
        self.fluctuation_density = fluctuation_number / area
        self.particle_density = condensate_density + self.fluctuation_density
        self.condensate_fraction = (
            condensate_density / self.particle_density
            if self.particle_density > 0.0
            else 0.0
        )
        self.kinetic_energy_density = self.kinetic_energy() / area
        self.condensate_mean_field_energy_density = condensate_mean_field
        self.condensate_normal_interaction_density = normal_shift
        self.condensate_anomalous_interaction_density = anomalous_shift
        self.fluctuation_quartic_interaction_density = fluctuation_quartic
        self.interaction_energy_density = (
            condensate_mean_field
            + normal_shift
            + anomalous_shift
            + fluctuation_quartic
        )
        self.energy_density = (
            self.kinetic_energy_density + self.interaction_energy_density
        )
        return self

    def shell_channel(self, energy, radial_width):
        r"""Contract one finite-width angular cLETTA shell channel."""
        radial_width = float(radial_width)
        if radial_width <= 0.0:
            raise ValueError("radial_width must be positive.")
        return self._shell_channel_from_amplitude(
            energy, np.sqrt(radial_width)
        )

    def effective_outer_generator(
        self, energy, *, counting_source=0.0, counting_weight=1.0
    ):
        r"""Return ``K_E = lim_(dE->0) (Phi_E(dE)-I) / dE``.

        Writing ``t=sqrt(dE)``, the vacuum-projected channel is even through
        the required order.  A centered second derivative in ``t`` therefore
        extracts the coefficient of ``dE`` without subtracting two nearly
        equal one-sided channels.
        """
        step = self.generator_step
        plus = self._shell_channel_from_amplitude(
            energy,
            step,
            counting_source=counting_source,
            counting_weight=counting_weight,
        )
        minus = self._shell_channel_from_amplitude(
            energy,
            -step,
            counting_source=counting_source,
            counting_weight=counting_weight,
        )
        identity = np.eye(self.outer_transfer_dim, dtype=np.complex128)
        return (plus + minus - 2.0 * identity) / (2.0 * step**2)

    def continuum_shell_channel(self, energy, radial_width):
        """Return the outer continuum propagator ``exp(dE * K_E)``."""
        return expm(
            float(radial_width)
            * self.replication_scale
            * self.effective_outer_generator(energy)
        )

    def outer_generator_derivative(self, energy, *, weight=1.0):
        """Differentiate the nested outer generator by a counting source."""
        step = self.generator_step
        plus = self._shell_channel_source_derivative_from_amplitude(
            energy, step, weight=weight
        )
        minus = self._shell_channel_source_derivative_from_amplitude(
            energy, -step, weight=weight
        )
        zero = self._shell_channel_source_derivative_from_amplitude(
            energy, 0.0, weight=weight
        )
        return (plus + minus - 2.0 * zero) / (2.0 * step**2)

    def boundary_vectors(self):
        outer = np.zeros(self.bond_dim, dtype=np.complex128)
        outer[0] = 1.0
        double = np.kron(outer, outer.conj())
        return double.copy(), double

    def contract(self, *, continuum=True):
        """Return ``(norm, final_outer_environment)``."""
        energies, radial_widths = self.contraction.radial_quadrature()
        left, environment = self.boundary_vectors()
        for energy, radial_width in zip(energies, radial_widths):
            if continuum:
                channel = self.continuum_shell_channel(energy, radial_width)
            else:
                channel = self.shell_channel(energy, radial_width)
            environment = channel @ environment
        value = complex(np.vdot(left, environment))
        return value, environment

    def norm(self, *, continuum=True):
        value, _ = self.contract(continuum=continuum)
        if abs(value.imag) > 1.0e-9 * max(1.0, abs(value.real)):
            raise FloatingPointError("nested hLETTA norm acquired an imaginary part.")
        return float(value.real)

    def additive_one_body_expectation(self, weight):
        r"""Return a continuum one-body expectation from nested sources."""
        if callable(weight):
            weight_function = weight
        else:
            constant = complex(weight)

            def weight_function(_energy):
                return constant

        energies, radial_widths = self.contraction.radial_quadrature()
        left, environment = self.boundary_vectors()
        tangent = np.zeros_like(environment)
        for energy, radial_width in zip(energies, radial_widths):
            generator = (
                self.replication_scale
                * self.effective_outer_generator(float(energy))
            )
            derivative = self.replication_scale * self.outer_generator_derivative(
                float(energy), weight=weight_function(float(energy))
            )
            propagator, propagator_derivative = expm_frechet(
                float(radial_width) * generator,
                float(radial_width) * derivative,
                compute_expm=True,
            )
            tangent = (
                propagator_derivative @ environment
                + propagator @ tangent
            )
            environment = propagator @ environment
        norm = complex(np.vdot(left, environment))
        derivative = complex(np.vdot(left, tangent))
        if abs(norm) <= np.finfo(float).tiny:
            raise FloatingPointError("the nested hLETTA norm is numerically zero.")
        return derivative / norm

    def particle_number(self):
        value = self.additive_one_body_expectation(1.0)
        return float(np.real_if_close(value).real)

    def kinetic_energy(self):
        value = self.additive_one_body_expectation(lambda energy: energy)
        return float(np.real_if_close(value).real)

    def gns_fixed_point(self, energy):
        r"""Return the dominant local transfer eigentriple at one energy.

        In the area-extensive limit the outer propagator is generated by
        ``A K_E``.  Away from a transfer-level crossing, its GNS state is
        determined by the dominant left and right eigenvectors of ``K_E``.
        They are normalized so that ``<l_E|r_E> = 1``.
        """
        energy = float(energy)
        if not (0.0 <= energy <= self.contraction.energy_cutoff):
            raise ValueError("energy lies outside the hLETTA cutoff.")
        cache = getattr(self, "_gns_fixed_point_cache", None)
        if cache is None:
            cache = {}
            self._gns_fixed_point_cache = cache
        key = round(energy, 14)
        if key in cache:
            return cache[key]

        generator = self.effective_outer_generator(energy)
        values, left_vectors, right_vectors = eig(
            generator, left=True, right=True
        )
        ordering = np.lexsort((np.abs(values.imag), -values.real))
        index = int(ordering[0])
        left = left_vectors[:, index]
        right = right_vectors[:, index]
        overlap = complex(np.vdot(left, right))
        if abs(overlap) <= 1.0e-12:
            raise FloatingPointError("dominant GNS transfer vectors are defective.")
        right = right / overlap
        other_real_parts = np.delete(values.real, index)
        gap = (
            float(values[index].real - np.max(other_real_parts))
            if other_real_parts.size
            else np.inf
        )
        result = (complex(values[index]), left, right, gap)
        cache[key] = result
        return result

    def gns_additive_one_body_density(self, weight=1.0):
        r"""Return an additive one-body observable directly per unit area."""
        if callable(weight):
            weight_function = weight
        else:
            constant = complex(weight)

            def weight_function(_energy):
                return constant

        energies, widths = self.contraction.radial_quadrature()
        total = 0.0j
        minimum_gap = np.inf
        for energy, width in zip(energies, widths):
            _, left, right, gap = self.gns_fixed_point(float(energy))
            derivative = self.outer_generator_derivative(
                float(energy), weight=weight_function(float(energy))
            )
            total += float(width) * np.vdot(left, derivative @ right)
            minimum_gap = min(minimum_gap, gap)
        self.minimum_gns_transfer_gap = float(minimum_gap)
        if abs(total.imag) > 1.0e-7 * max(1.0, abs(total.real)):
            raise FloatingPointError("GNS one-body density acquired an imaginary part.")
        return float(total.real)

    def gns_particle_density(self):
        return self.gns_additive_one_body_density(1.0)

    def gns_kinetic_energy_density(self):
        return self.gns_additive_one_body_density(lambda energy: energy)

    def _gns_local_insertion(self, energy, insertion):
        _, left, right, _ = self.gns_fixed_point(float(energy))
        return complex(np.vdot(left, insertion @ right))

    def gns_local_density(self, energy, theta=0.0):
        insertion = self.effective_field_insertion(
            float(energy), float(theta), "density"
        ) / self.replication_scale
        return self._gns_local_insertion(energy, insertion)

    def gns_local_antipodal_pair(
        self, energy, theta=0.0, *, operator="annihilation"
    ):
        operator = str(operator).lower()
        if operator not in {"annihilation", "creation"}:
            raise ValueError("operator must be annihilation or creation.")
        representative = float(theta) % np.pi
        insertion = self.effective_pair_insertion(
            float(energy),
            representative,
            operator,
            representative + np.pi,
            operator,
        ) / self.replication_scale
        return self._gns_local_insertion(energy, insertion)

    def gns_antipodal_pair_density(self, weight=1.0, *, angular_points=3):
        r"""Return ``integral dE integral_0^pi dtheta w(E) <b_k b_-k>/A``."""
        if callable(weight):
            weight_function = weight
        else:
            constant = complex(weight)

            def weight_function(_energy):
                return constant

        angular_points = int(angular_points)
        if angular_points < 2:
            raise ValueError("angular_points must be at least two.")
        angular_nodes, angular_weights = leggauss(angular_points)
        angles = 0.5 * np.pi * (angular_nodes + 1.0)
        angle_widths = 0.5 * np.pi * angular_weights
        energies, energy_widths = self.contraction.radial_quadrature()
        total = 0.0j
        for energy, energy_width in zip(energies, energy_widths):
            for theta, theta_width in zip(angles, angle_widths):
                total += (
                    float(energy_width)
                    * float(theta_width)
                    * complex(weight_function(float(energy)))
                    * self.gns_local_antipodal_pair(
                        float(energy), float(theta)
                    )
                )
        return total

    def gns_normal_ordered_density_structure(
        self,
        momentum_transfer,
        *,
        kinetic_prefactor=1.0,
        radial_points=3,
        angular_points=4,
    ):
        r"""Return the momentum-conserving GNS four-field structure per area.

        In the dominant-transfer state, distinct energy shells factorize in
        the area limit.  For nonzero transfer this leaves the normal-exchange
        and antipodal-pair channels

        $$
        S_4(q)=\int_k\left[n_k n_{k+q}+m_k m_{k+q}^*\right].
        $$
        """
        momentum_transfer = float(momentum_transfer)
        kinetic_prefactor = float(kinetic_prefactor)
        radial_points = int(radial_points)
        angular_points = int(angular_points)
        if momentum_transfer < 0.0 or kinetic_prefactor <= 0.0:
            raise ValueError(
                "momentum_transfer must be non-negative and kinetic_prefactor positive."
            )
        if radial_points < 2 or angular_points < 3:
            raise ValueError(
                "radial_points must be at least two and angular_points at least three."
            )
        cutoff = self.contraction.energy_cutoff
        momentum_cutoff = np.sqrt(cutoff / kinetic_prefactor)
        radial_nodes, radial_weights = leggauss(radial_points)
        energies = 0.5 * cutoff * (radial_nodes + 1.0)
        energy_weights = 0.5 * cutoff * radial_weights
        angular_nodes, angular_weights = leggauss(angular_points)
        angles = np.pi * (angular_nodes + 1.0)
        angle_weights = np.pi * angular_weights
        transfer_vector = np.array([momentum_transfer, 0.0])
        local_cache = {}

        def local_values(energy, theta):
            key = (round(float(energy), 13), round(float(theta) % np.pi, 13))
            if key not in local_cache:
                density = self.gns_local_density(energy, theta)
                annihilation_pair = self.gns_local_antipodal_pair(
                    energy, theta, operator="annihilation"
                )
                creation_pair = self.gns_local_antipodal_pair(
                    energy, theta, operator="creation"
                )
                local_cache[key] = (
                    density,
                    annihilation_pair,
                    creation_pair,
                )
            return local_cache[key]

        total = 0.0j
        for energy, energy_weight in zip(energies, energy_weights):
            momentum = np.sqrt(float(energy) / kinetic_prefactor)
            for theta, theta_weight in zip(angles, angle_weights):
                vector = np.array(
                    [momentum * np.cos(theta), momentum * np.sin(theta)]
                )
                shifted = vector + transfer_vector
                shifted_norm, shifted_theta = self._polar_coordinates(shifted)
                if shifted_norm >= momentum_cutoff:
                    continue
                shifted_energy = kinetic_prefactor * shifted_norm**2
                density, annihilation_pair, _ = local_values(
                    float(energy), float(theta)
                )
                shifted_density, _, shifted_creation_pair = local_values(
                    shifted_energy, shifted_theta
                )
                total += (
                    float(energy_weight)
                    * float(theta_weight)
                    * (
                        density * shifted_density
                        + annihilation_pair * shifted_creation_pair
                    )
                )
        if abs(total.imag) > 2.0e-6 * max(1.0, abs(total.real)):
            raise FloatingPointError(
                "GNS density structure acquired an imaginary part."
            )
        return float(total.real)

    def evaluate_condensate_shifted_gns_rank_one(
        self,
        channel,
        *,
        condensate_density,
        kinetic_prefactor=1.0,
        structure_radial_points=3,
        structure_angular_points=4,
        pair_angular_points=3,
    ):
        r"""Evaluate the condensate-shifted Hamiltonian directly in the GNS state."""
        if not isinstance(channel, RankOneDensityTransferChannel2D):
            raise TypeError("channel must be RankOneDensityTransferChannel2D.")
        if not getattr(self, "parity_even", False):
            raise ValueError(
                "condensate shifting requires a parity-even fluctuation ansatz."
            )
        condensate_density = float(condensate_density)
        kinetic_prefactor = float(kinetic_prefactor)
        if condensate_density < 0.0 or kinetic_prefactor <= 0.0:
            raise ValueError(
                "condensate_density must be non-negative and kinetic_prefactor positive."
            )
        momenta, _ = channel.radial_quadrature()
        structure = np.asarray(
            [
                self.gns_normal_ordered_density_structure(
                    momentum,
                    kinetic_prefactor=kinetic_prefactor,
                    radial_points=structure_radial_points,
                    angular_points=structure_angular_points,
                )
                for momentum in momenta
            ]
        )
        profile_zero = complex(
            np.asarray(channel.radial_profile(0.0), dtype=np.complex128).item()
        )
        potential_zero = abs(profile_zero) ** 2

        def potential_at_energy(energy):
            momentum = np.sqrt(float(energy) / kinetic_prefactor)
            profile = complex(
                np.asarray(
                    channel.radial_profile(momentum), dtype=np.complex128
                ).item()
            )
            return abs(profile) ** 2

        fluctuation_density = self.gns_particle_density()
        normal_shift = condensate_density * self.gns_additive_one_body_density(
            lambda energy: potential_zero + potential_at_energy(energy)
        )
        half_plane_pair = self.gns_antipodal_pair_density(
            potential_at_energy, angular_points=pair_angular_points
        )
        anomalous_shift = 2.0 * condensate_density * float(
            half_plane_pair.real
        )
        condensate_mean_field = (
            0.5 * potential_zero * condensate_density**2
        )
        fluctuation_quartic = (
            channel.normal_ordered_interaction_energy_density(structure)
        )

        self.density_transfer_momenta = momenta
        self.density_transfer_structure = structure
        self.condensate_density = condensate_density
        self.fluctuation_density = fluctuation_density
        self.particle_density = condensate_density + fluctuation_density
        self.condensate_fraction = (
            condensate_density / self.particle_density
            if self.particle_density > 0.0
            else 0.0
        )
        self.kinetic_energy_density = self.gns_kinetic_energy_density()
        self.condensate_mean_field_energy_density = condensate_mean_field
        self.condensate_normal_interaction_density = normal_shift
        self.condensate_anomalous_interaction_density = anomalous_shift
        self.fluctuation_quartic_interaction_density = fluctuation_quartic
        self.interaction_energy_density = (
            condensate_mean_field
            + normal_shift
            + anomalous_shift
            + fluctuation_quartic
        )
        self.energy_density = (
            self.kinetic_energy_density + self.interaction_energy_density
        )
        self.thermodynamic_energy_density = self.energy_density
        self.area_drift = 0.0
        self.asymptotic_area_drift = 0.0
        self.thermodynamic_extrapolation_drift = 0.0
        self.thermodynamic_valid = True
        return self


def fixed_density_nested_hletta_state(
    contraction,
    *,
    target_density,
    area=1.0,
    outer_gap=0.3,
    tie_ratio=1.0 / 3.0,
    angular_memory_decay=0.7,
    radial_decay=0.9,
    field_phase=0.0,
    replication_scale=1.0,
    generator_step=1.0e-3,
    density_tolerance=1.0e-8,
):
    r"""Build the minimal nested state and solve its field scale at fixed density.

    The gauge-fixed scalar parameterization is

    $$
    Q=\operatorname{diag}(0,-\Delta_Q),\qquad
    R=\alpha\begin{pmatrix}0&1\\2/3&0\end{pmatrix},
    $$

    $$
    S=\alpha\eta\begin{pmatrix}0&1\\3/4&0\end{pmatrix}.
    $$

    ``alpha`` is solved so that ``particle_number / area == target_density``.
    """
    if not isinstance(contraction, HierarchicalShellContraction):
        raise TypeError("contraction must be HierarchicalShellContraction.")
    target_density = float(target_density)
    area = float(area)
    outer_gap = float(outer_gap)
    tie_ratio = float(tie_ratio)
    angular_memory_decay = float(angular_memory_decay)
    radial_decay = float(radial_decay)
    field_phase = float(field_phase)
    replication_scale = float(replication_scale)
    density_tolerance = float(density_tolerance)
    if target_density < 0.0 or area <= 0.0:
        raise ValueError("target_density must be non-negative and area positive.")
    if outer_gap <= 0.0 or not np.isfinite(tie_ratio):
        raise ValueError("outer_gap must be positive and tie_ratio finite.")
    if angular_memory_decay <= 0.0 or radial_decay < 0.0:
        raise ValueError(
            "angular_memory_decay must be positive and radial_decay non-negative."
        )
    if not np.isfinite(field_phase):
        raise ValueError("field_phase must be finite.")
    if not np.isfinite(replication_scale) or replication_scale <= 0.0:
        raise ValueError("replication_scale must be finite and positive.")
    target_number = target_density * area
    q_matrix = np.diag([0.0, -outer_gap])
    r_pattern = np.array([[0.0, 1.0], [2.0 / 3.0, 0.0]])
    tie_pattern = np.array([[0.0, 1.0], [3.0 / 4.0, 0.0]])

    def state_at_amplitude(amplitude):
        state = D2M1NestedCLETTA2D(
            contraction=contraction,
            q_matrix=q_matrix,
            r_matrix=float(amplitude) * r_pattern,
            tie_matrix=float(amplitude) * tie_ratio * tie_pattern,
            angular_memory_decay=angular_memory_decay,
            radial_decay=radial_decay,
            field_phase=field_phase,
            replication_scale=replication_scale,
            generator_step=generator_step,
        )
        state.field_amplitude = float(amplitude)
        state.outer_gap = outer_gap
        state.tie_ratio = tie_ratio
        state.target_density = target_density
        state.area = area
        state.replication_area = replication_scale
        state.parity_even = True
        return state

    if target_number == 0.0:
        state = state_at_amplitude(0.0)
        state.particle_density = 0.0
        return state

    upper = 1.0
    upper_number = state_at_amplitude(upper).particle_number()
    while upper_number < target_number and upper < 128.0:
        upper *= 2.0
        upper_number = state_at_amplitude(upper).particle_number()
    if not np.isfinite(upper_number) or upper_number < target_number:
        raise RuntimeError(
            "target density is not reachable within the stable amplitude bracket."
        )

    amplitude = brentq(
        lambda value: state_at_amplitude(value).particle_number() - target_number,
        0.0,
        upper,
        xtol=density_tolerance,
        rtol=4.0 * np.finfo(float).eps,
        maxiter=100,
    )
    state = state_at_amplitude(amplitude)
    state.particle_density = state.particle_number() / area
    if abs(state.particle_density - target_density) > max(
        10.0 * density_tolerance, 2.0e-7 * target_density
    ):
        raise FloatingPointError("fixed-density solve did not reach its target.")
    return state


def fixed_density_gns_nested_hletta_state(
    contraction,
    *,
    target_density,
    outer_gap=0.3,
    tie_ratio=1.0 / 3.0,
    angular_memory_decay=0.7,
    radial_decay=0.9,
    field_phase=0.0,
    generator_step=1.0e-3,
    density_tolerance=1.0e-8,
):
    r"""Solve the fluctuation field scale at fixed GNS density.

    Unlike :func:`fixed_density_nested_hletta_state`, this routine never
    introduces a finite particle number or vacuum outer boundary into the
    density constraint.  It solves ``gns_particle_density() == target_density``
    directly from the dominant transfer fixed points.
    """
    if not isinstance(contraction, HierarchicalShellContraction):
        raise TypeError("contraction must be HierarchicalShellContraction.")
    target_density = float(target_density)
    outer_gap = float(outer_gap)
    tie_ratio = float(tie_ratio)
    angular_memory_decay = float(angular_memory_decay)
    radial_decay = float(radial_decay)
    field_phase = float(field_phase)
    density_tolerance = float(density_tolerance)
    if target_density < 0.0:
        raise ValueError("target_density must be non-negative.")
    if outer_gap <= 0.0 or not np.isfinite(tie_ratio):
        raise ValueError("outer_gap must be positive and tie_ratio finite.")
    if angular_memory_decay <= 0.0 or radial_decay < 0.0:
        raise ValueError(
            "angular_memory_decay must be positive and radial_decay non-negative."
        )
    if not np.isfinite(field_phase):
        raise ValueError("field_phase must be finite.")

    q_matrix = np.diag([0.0, -outer_gap])
    r_pattern = np.array([[0.0, 1.0], [2.0 / 3.0, 0.0]])
    tie_pattern = np.array([[0.0, 1.0], [3.0 / 4.0, 0.0]])

    def state_at_amplitude(amplitude):
        state = D2M1NestedCLETTA2D(
            contraction=contraction,
            q_matrix=q_matrix,
            r_matrix=float(amplitude) * r_pattern,
            tie_matrix=float(amplitude) * tie_ratio * tie_pattern,
            angular_memory_decay=angular_memory_decay,
            radial_decay=radial_decay,
            field_phase=field_phase,
            replication_scale=1.0,
            generator_step=generator_step,
        )
        state.field_amplitude = float(amplitude)
        state.outer_gap = outer_gap
        state.tie_ratio = tie_ratio
        state.target_density = target_density
        state.parity_even = True
        state.uses_gns_boundaries = True
        return state

    if target_density == 0.0:
        state = state_at_amplitude(0.0)
        state.particle_density = 0.0
        return state

    upper = 1.0
    upper_density = state_at_amplitude(upper).gns_particle_density()
    while upper_density < target_density and upper < 128.0:
        upper *= 2.0
        upper_density = state_at_amplitude(upper).gns_particle_density()
    if not np.isfinite(upper_density) or upper_density < target_density:
        raise RuntimeError(
            "target GNS density is not reachable within the stable amplitude bracket."
        )
    amplitude = brentq(
        lambda value: (
            state_at_amplitude(value).gns_particle_density() - target_density
        ),
        0.0,
        upper,
        xtol=density_tolerance,
        rtol=4.0 * np.finfo(float).eps,
        maxiter=100,
    )
    state = state_at_amplitude(amplitude)
    state.particle_density = state.gns_particle_density()
    if abs(state.particle_density - target_density) > max(
        10.0 * density_tolerance, 2.0e-7 * max(target_density, 1.0)
    ):
        raise FloatingPointError("fixed GNS density solve did not reach its target.")
    return state


def optimize_nested_hletta_fixed_density(
    model,
    *,
    target_density=None,
    area=1.0,
    energy_cutoff=1.0,
    radial_points=8,
    angular_points=8,
    channel_points=8,
    structure_radial_points=2,
    structure_angular_points=3,
    propagation_points=2,
    outer_gap=0.3,
    angular_memory_decay=0.7,
    initial_tie_ratio=1.0 / 3.0,
    initial_radial_decay=0.9,
    tie_ratio_bounds=(1.0e-3, 3.0),
    radial_decay_bounds=(0.1, 5.0),
    extensivity_tolerance=0.05,
    maxiter=30,
):
    r"""Optimize the first fixed-density ``D=2, M=1`` interacting ansatz.

    The density-fixing amplitude is eliminated exactly.  The free shape
    variables are the positive tie ratio and radial decay.
    """
    if not isinstance(model, GaussianPotentialBoseGas2D):
        raise TypeError("model must be GaussianPotentialBoseGas2D.")
    if target_density is None:
        target_density = model.density
    target_density = float(target_density)
    area = float(area)
    tie_ratio_bounds = tuple(float(value) for value in tie_ratio_bounds)
    radial_decay_bounds = tuple(float(value) for value in radial_decay_bounds)
    if not (
        0.0 < tie_ratio_bounds[0] < tie_ratio_bounds[1]
        and 0.0 < radial_decay_bounds[0] < radial_decay_bounds[1]
    ):
        raise ValueError("optimizer bounds must be ordered and positive.")
    contraction = HierarchicalShellContraction(
        energy_cutoff=energy_cutoff,
        radial_points=radial_points,
        angular_points=angular_points,
    )
    momentum_cutoff = 2.0 * np.sqrt(
        float(energy_cutoff) / model.kinetic_prefactor
    )
    channel = RankOneDensityTransferChannel2D(
        radial_profile=model.density_transfer_profile,
        momentum_cutoff=momentum_cutoff,
        radial_points=channel_points,
    )
    history = []
    best = [np.inf, None]

    def evaluate(parameters):
        tie_ratio = float(np.exp(parameters[0]))
        radial_decay = float(np.exp(parameters[1]))
        try:
            state = fixed_density_nested_hletta_state(
                contraction,
                target_density=target_density,
                area=area,
                outer_gap=outer_gap,
                tie_ratio=tie_ratio,
                angular_memory_decay=angular_memory_decay,
                radial_decay=radial_decay,
            )
            state.evaluate_rank_one_density_transfer(
                channel,
                kinetic_prefactor=model.kinetic_prefactor,
                area=area,
                structure_radial_points=structure_radial_points,
                structure_angular_points=structure_angular_points,
                propagation_points=propagation_points,
            )
            energy = float(state.energy_density)
        except (FloatingPointError, RuntimeError, ValueError, OverflowError):
            energy = 1.0e12
            state = None
        history.append((tie_ratio, radial_decay, energy))
        if np.isfinite(energy) and energy < best[0]:
            best[:] = [energy, state]
        return energy

    initial = np.log([float(initial_tie_ratio), float(initial_radial_decay)])
    initial_energy = evaluate(initial)
    result = minimize(
        evaluate,
        initial,
        method="Nelder-Mead",
        bounds=[
            tuple(np.log(tie_ratio_bounds)),
            tuple(np.log(radial_decay_bounds)),
        ],
        options={
            "maxiter": int(maxiter),
            "xatol": 2.0e-3,
            "fatol": 2.0e-6,
        },
    )
    if best[1] is None:
        raise RuntimeError("all fixed-density hLETTA trials failed.")
    state = best[1]
    state.initial_energy_density = float(initial_energy)
    state.optimization_history = history
    tie_ratio = state.tie_ratio
    radial_decay = state.radial_decay
    boundary_limited = bool(
        tie_ratio <= 1.01 * tie_ratio_bounds[0]
        or tie_ratio >= 0.99 * tie_ratio_bounds[1]
        or radial_decay <= 1.01 * radial_decay_bounds[0]
        or radial_decay >= 0.99 * radial_decay_bounds[1]
    )
    validation_state = fixed_density_nested_hletta_state(
        contraction,
        target_density=target_density,
        area=2.0 * area,
        outer_gap=outer_gap,
        tie_ratio=tie_ratio,
        angular_memory_decay=angular_memory_decay,
        radial_decay=radial_decay,
    )
    validation_state.evaluate_rank_one_density_transfer(
        channel,
        kinetic_prefactor=model.kinetic_prefactor,
        area=2.0 * area,
        structure_radial_points=structure_radial_points,
        structure_angular_points=structure_angular_points,
        propagation_points=propagation_points,
    )
    state.area_doubled_energy_density = validation_state.energy_density
    state.area_drift = abs(
        validation_state.energy_density - state.energy_density
    ) / max(abs(state.energy_density), 1.0e-12)
    state.boundary_limited = boundary_limited
    state.thermodynamic_valid = bool(
        not boundary_limited and state.area_drift <= float(extensivity_tolerance)
    )
    state.success = bool(
        result.success
        and np.isfinite(best[0])
        and state.thermodynamic_valid
    )
    messages = [str(result.message)]
    if boundary_limited:
        messages.append("shape optimum reached a parameter bound")
    if state.area_drift > float(extensivity_tolerance):
        messages.append(
            "energy density failed the area-doubling extensivity check"
        )
    state.message = "; ".join(messages)
    state.optimized_parameters = {
        "field_amplitude": state.field_amplitude,
        "outer_gap": state.outer_gap,
        "tie_ratio": state.tie_ratio,
        "angular_memory_decay": state.angular_memory_decay,
        "radial_decay": state.radial_decay,
    }
    return state


def optimize_condensate_nested_hletta_fixed_density(
    model,
    *,
    target_density=None,
    area=16.0,
    energy_cutoff=1.0,
    radial_points=8,
    angular_points=8,
    channel_points=8,
    structure_radial_points=2,
    structure_angular_points=3,
    pair_angular_points=3,
    propagation_points=2,
    outer_gap=0.3,
    angular_memory_decay=0.7,
    initial_condensate_fraction=None,
    initial_tie_ratio=1.0 / 3.0,
    initial_radial_decay=0.9,
    condensate_fraction_bounds=(1.0e-4, 0.9999),
    tie_ratio_bounds=(1.0e-3, 3.0),
    radial_decay_bounds=(0.1, 5.0),
    extensivity_tolerance=0.05,
    maxiter=30,
):
    r"""Optimize ``n0`` and a parity-even nested hLETTA fluctuation state.

    For each trial condensate fraction ``f0``, the hLETTA field amplitude is
    eliminated by the hard constraint

    $$
    n_{\rm ex}=(1-f_0)n,\qquad n_0=f_0n.
    $$

    The global fluctuation-field phase is fixed to ``pi/2`` relative to the
    real condensate.  This selects the energy-lowering sign of the antipodal
    anomalous expectation for the real ``D=2, M=1`` parameterization.  The
    pure-condensate endpoint is evaluated explicitly.  ``area`` multiplies
    the outer transfer generator and is therefore the number of replicated
    real-space area units, rather than a target-number label.  The selected
    state is audited at ``area``, ``2*area``, and ``4*area``; two successive
    ``1/A`` extrapolations provide the bulk energy and its convergence error.
    """
    if not isinstance(model, GaussianPotentialBoseGas2D):
        raise TypeError("model must be GaussianPotentialBoseGas2D.")
    if target_density is None:
        target_density = model.density
    target_density = float(target_density)
    area = float(area)
    if target_density <= 0.0 or area <= 0.0:
        raise ValueError("target_density and area must be positive.")
    condensate_fraction_bounds = tuple(
        float(value) for value in condensate_fraction_bounds
    )
    tie_ratio_bounds = tuple(float(value) for value in tie_ratio_bounds)
    radial_decay_bounds = tuple(float(value) for value in radial_decay_bounds)
    if not (
        0.0 <= condensate_fraction_bounds[0]
        < condensate_fraction_bounds[1]
        < 1.0
    ):
        raise ValueError(
            "condensate_fraction_bounds must lie in [0, 1) and be ordered."
        )
    if not (
        0.0 < tie_ratio_bounds[0] < tie_ratio_bounds[1]
        and 0.0 < radial_decay_bounds[0] < radial_decay_bounds[1]
    ):
        raise ValueError("shape bounds must be ordered and positive.")

    contraction = HierarchicalShellContraction(
        energy_cutoff=energy_cutoff,
        radial_points=radial_points,
        angular_points=angular_points,
    )
    momentum_cutoff = 2.0 * np.sqrt(
        float(energy_cutoff) / model.kinetic_prefactor
    )
    channel = RankOneDensityTransferChannel2D(
        radial_profile=model.density_transfer_profile,
        momentum_cutoff=momentum_cutoff,
        radial_points=channel_points,
    )
    history = []
    best = [np.inf, None, None]

    def build_and_evaluate(condensate_fraction, tie_ratio, radial_decay, trial_area):
        condensate_fraction = float(condensate_fraction)
        fluctuation_density = (1.0 - condensate_fraction) * target_density
        state = fixed_density_nested_hletta_state(
            contraction,
            target_density=fluctuation_density,
            area=trial_area,
            outer_gap=outer_gap,
            tie_ratio=tie_ratio,
            angular_memory_decay=angular_memory_decay,
            radial_decay=radial_decay,
            field_phase=0.5 * np.pi,
            replication_scale=trial_area,
        )
        state.evaluate_condensate_shifted_rank_one(
            channel,
            condensate_density=condensate_fraction * target_density,
            kinetic_prefactor=model.kinetic_prefactor,
            area=trial_area,
            structure_radial_points=structure_radial_points,
            structure_angular_points=structure_angular_points,
            pair_angular_points=pair_angular_points,
            propagation_points=propagation_points,
            include_smooth_structure=False,
        )
        return state

    def evaluate(parameters):
        condensate_fraction = float(parameters[0])
        tie_ratio = float(np.exp(parameters[1]))
        radial_decay = float(np.exp(parameters[2]))
        try:
            state = build_and_evaluate(
                condensate_fraction, tie_ratio, radial_decay, area
            )
            doubled_state = build_and_evaluate(
                condensate_fraction, tie_ratio, radial_decay, 2.0 * area
            )
            energy = float(
                2.0 * doubled_state.energy_density - state.energy_density
            )
        except (FloatingPointError, RuntimeError, ValueError, OverflowError):
            state = None
            doubled_state = None
            energy = 1.0e12
        history.append(
            (
                condensate_fraction,
                tie_ratio,
                radial_decay,
                energy,
                None if state is None else float(state.energy_density),
                None
                if doubled_state is None
                else float(doubled_state.energy_density),
            )
        )
        if np.isfinite(energy) and energy < best[0]:
            best[:] = [energy, state, doubled_state]
        return energy

    if initial_condensate_fraction is None:
        depletion_fraction = model.depletion_density() / target_density
        initial_condensate_fraction = 1.0 - depletion_fraction
    initial_condensate_fraction = float(
        np.clip(
            initial_condensate_fraction,
            condensate_fraction_bounds[0],
            condensate_fraction_bounds[1],
        )
    )
    initial = np.array(
        [
            initial_condensate_fraction,
            np.log(float(initial_tie_ratio)),
            np.log(float(initial_radial_decay)),
        ]
    )
    initial_energy = evaluate(initial)
    result = minimize(
        evaluate,
        initial,
        method="Nelder-Mead",
        bounds=[
            condensate_fraction_bounds,
            tuple(np.log(tie_ratio_bounds)),
            tuple(np.log(radial_decay_bounds)),
        ],
        options={
            "maxiter": int(maxiter),
            "xatol": 2.0e-3,
            "fatol": 2.0e-6,
        },
    )

    pure_state = build_and_evaluate(
        1.0,
        float(initial_tie_ratio),
        float(initial_radial_decay),
        area,
    )
    pure_doubled_state = build_and_evaluate(
        1.0,
        float(initial_tie_ratio),
        float(initial_radial_decay),
        2.0 * area,
    )
    pure_bulk_energy = float(
        2.0 * pure_doubled_state.energy_density - pure_state.energy_density
    )
    history.append(
        (
            1.0,
            float(initial_tie_ratio),
            float(initial_radial_decay),
            pure_bulk_energy,
            float(pure_state.energy_density),
            float(pure_doubled_state.energy_density),
        )
    )
    if pure_bulk_energy < best[0]:
        best[:] = [pure_bulk_energy, pure_state, pure_doubled_state]
    if best[1] is None:
        raise RuntimeError("all condensate-plus-hLETTA trials failed.")

    state = best[1]
    state.initial_energy_density = float(initial_energy)
    state.optimization_energy_density = float(best[0])
    state.pure_condensate_energy_density = float(pure_state.energy_density)
    state.optimization_history = history
    condensate_fraction = float(state.condensate_fraction)
    tie_ratio = float(state.tie_ratio)
    radial_decay = float(state.radial_decay)
    condensate_endpoint = bool(
        condensate_fraction <= 1.0e-10
        or condensate_fraction >= 1.0 - 1.0e-10
    )
    shape_boundary_limited = bool(
        tie_ratio <= 1.01 * tie_ratio_bounds[0]
        or tie_ratio >= 0.99 * tie_ratio_bounds[1]
        or radial_decay <= 1.01 * radial_decay_bounds[0]
        or radial_decay >= 0.99 * radial_decay_bounds[1]
    )

    validation_state = best[2]
    second_validation_state = build_and_evaluate(
        condensate_fraction,
        tie_ratio,
        radial_decay,
        4.0 * area,
    )
    state.area_doubled_energy_density = validation_state.energy_density
    state.area_quadrupled_energy_density = (
        second_validation_state.energy_density
    )
    state.area_drift = abs(
        validation_state.energy_density - state.energy_density
    ) / max(abs(state.energy_density), 1.0e-12)
    state.asymptotic_area_drift = abs(
        second_validation_state.energy_density
        - validation_state.energy_density
    ) / max(abs(second_validation_state.energy_density), 1.0e-12)
    first_extrapolation = (
        2.0 * validation_state.energy_density - state.energy_density
    )
    second_extrapolation = (
        2.0 * second_validation_state.energy_density
        - validation_state.energy_density
    )
    state.thermodynamic_energy_density = float(second_extrapolation)
    state.thermodynamic_extrapolation_drift = abs(
        second_extrapolation - first_extrapolation
    ) / max(abs(second_extrapolation), 1.0e-12)
    state.condensate_endpoint = condensate_endpoint
    state.shape_boundary_limited = shape_boundary_limited
    state.boundary_limited = bool(
        condensate_endpoint or shape_boundary_limited
    )
    state.density_constraint_error = state.particle_density - target_density
    state.thermodynamic_valid = bool(
        state.asymptotic_area_drift <= float(extensivity_tolerance)
        and state.thermodynamic_extrapolation_drift
        <= float(extensivity_tolerance)
    )
    state.success = bool(
        result.success
        and np.isfinite(best[0])
        and state.thermodynamic_valid
    )
    messages = [str(result.message)]
    if condensate_endpoint:
        messages.append("the variational minimum is a condensate endpoint")
    if shape_boundary_limited:
        messages.append("fluctuation shape reached a parameter bound")
    if not state.thermodynamic_valid:
        messages.append(
            "energy density failed the A, 2A, 4A bulk extrapolation"
        )
    state.message = "; ".join(messages)
    state.optimized_parameters = {
        "condensate_density": state.condensate_density,
        "fluctuation_density": state.fluctuation_density,
        "condensate_fraction": state.condensate_fraction,
        "field_amplitude": state.field_amplitude,
        "field_phase": state.field_phase,
        "replication_area": state.replication_scale,
        "outer_gap": state.outer_gap,
        "tie_ratio": state.tie_ratio,
        "angular_memory_decay": state.angular_memory_decay,
        "radial_decay": state.radial_decay,
    }
    return state


def optimize_condensate_gns_hletta_fixed_density(
    model,
    *,
    target_density=None,
    energy_cutoff=1.0,
    radial_points=8,
    angular_points=8,
    channel_points=8,
    structure_radial_points=3,
    structure_angular_points=4,
    pair_angular_points=3,
    outer_gap=0.3,
    angular_memory_decay=0.7,
    initial_condensate_fraction=None,
    initial_tie_ratio=1.0 / 3.0,
    initial_radial_decay=0.9,
    condensate_fraction_bounds=(1.0e-4, 0.9999),
    tie_ratio_bounds=(1.0e-3, 3.0),
    radial_decay_bounds=(0.1, 5.0),
    gns_gap_tolerance=1.0e-7,
    maxiter=30,
):
    r"""Optimize the condensate and hLETTA fluctuations in the GNS limit.

    All densities are derivatives of the dominant local transfer eigenvalue;
    vacuum outer vectors and a finite real-space area never enter this
    functional.  Consequently the returned energy is directly per unit area
    and has no area-doubling drift.
    """
    if not isinstance(model, GaussianPotentialBoseGas2D):
        raise TypeError("model must be GaussianPotentialBoseGas2D.")
    if target_density is None:
        target_density = model.density
    target_density = float(target_density)
    if target_density <= 0.0:
        raise ValueError("target_density must be positive.")
    condensate_fraction_bounds = tuple(
        float(value) for value in condensate_fraction_bounds
    )
    tie_ratio_bounds = tuple(float(value) for value in tie_ratio_bounds)
    radial_decay_bounds = tuple(float(value) for value in radial_decay_bounds)
    if not (
        0.0 <= condensate_fraction_bounds[0]
        < condensate_fraction_bounds[1]
        < 1.0
    ):
        raise ValueError(
            "condensate_fraction_bounds must lie in [0, 1) and be ordered."
        )
    if not (
        0.0 < tie_ratio_bounds[0] < tie_ratio_bounds[1]
        and 0.0 < radial_decay_bounds[0] < radial_decay_bounds[1]
    ):
        raise ValueError("shape bounds must be ordered and positive.")

    contraction = HierarchicalShellContraction(
        energy_cutoff=energy_cutoff,
        radial_points=radial_points,
        angular_points=angular_points,
    )
    momentum_cutoff = 2.0 * np.sqrt(
        float(energy_cutoff) / model.kinetic_prefactor
    )
    channel = RankOneDensityTransferChannel2D(
        radial_profile=model.density_transfer_profile,
        momentum_cutoff=momentum_cutoff,
        radial_points=channel_points,
    )
    history = []
    best = [np.inf, None]

    def build_and_evaluate(condensate_fraction, tie_ratio, radial_decay):
        fluctuation_density = (
            1.0 - float(condensate_fraction)
        ) * target_density
        state = fixed_density_gns_nested_hletta_state(
            contraction,
            target_density=fluctuation_density,
            outer_gap=outer_gap,
            tie_ratio=tie_ratio,
            angular_memory_decay=angular_memory_decay,
            radial_decay=radial_decay,
            field_phase=0.5 * np.pi,
        )
        state.evaluate_condensate_shifted_gns_rank_one(
            channel,
            condensate_density=float(condensate_fraction) * target_density,
            kinetic_prefactor=model.kinetic_prefactor,
            structure_radial_points=structure_radial_points,
            structure_angular_points=structure_angular_points,
            pair_angular_points=pair_angular_points,
        )
        return state

    def evaluate(parameters):
        condensate_fraction = float(parameters[0])
        tie_ratio = float(np.exp(parameters[1]))
        radial_decay = float(np.exp(parameters[2]))
        try:
            state = build_and_evaluate(
                condensate_fraction, tie_ratio, radial_decay
            )
            energy = float(state.energy_density)
        except (FloatingPointError, RuntimeError, ValueError, OverflowError):
            state = None
            energy = 1.0e12
        history.append(
            (condensate_fraction, tie_ratio, radial_decay, energy)
        )
        if np.isfinite(energy) and energy < best[0]:
            best[:] = [energy, state]
        return energy

    if initial_condensate_fraction is None:
        initial_condensate_fraction = (
            1.0 - model.depletion_density() / target_density
        )
    initial_condensate_fraction = float(
        np.clip(
            initial_condensate_fraction,
            condensate_fraction_bounds[0],
            condensate_fraction_bounds[1],
        )
    )
    initial = np.array(
        [
            initial_condensate_fraction,
            np.log(float(initial_tie_ratio)),
            np.log(float(initial_radial_decay)),
        ]
    )
    initial_energy = evaluate(initial)
    result = minimize(
        evaluate,
        initial,
        method="Nelder-Mead",
        bounds=[
            condensate_fraction_bounds,
            tuple(np.log(tie_ratio_bounds)),
            tuple(np.log(radial_decay_bounds)),
        ],
        options={
            "maxiter": int(maxiter),
            "xatol": 2.0e-3,
            "fatol": 2.0e-6,
        },
    )
    pure_state = build_and_evaluate(
        1.0, float(initial_tie_ratio), float(initial_radial_decay)
    )
    history.append(
        (1.0, float(initial_tie_ratio), float(initial_radial_decay), pure_state.energy_density)
    )
    if pure_state.energy_density < best[0]:
        best[:] = [float(pure_state.energy_density), pure_state]
    if best[1] is None:
        raise RuntimeError("all condensate-plus-GNS-hLETTA trials failed.")

    state = best[1]
    state.initial_energy_density = float(initial_energy)
    state.pure_condensate_energy_density = float(pure_state.energy_density)
    state.optimization_history = history
    state.optimization_energy_density = float(best[0])
    state.area_doubled_energy_density = state.energy_density
    state.area_quadrupled_energy_density = state.energy_density
    state.area_drift = 0.0
    state.asymptotic_area_drift = 0.0
    state.thermodynamic_extrapolation_drift = 0.0
    state.thermodynamic_energy_density = state.energy_density
    state.uses_gns_boundaries = True
    state.density_constraint_error = state.particle_density - target_density
    condensate_endpoint = bool(
        state.condensate_fraction <= 1.0e-10
        or state.condensate_fraction >= 1.0 - 1.0e-10
    )
    shape_boundary_limited = bool(
        state.tie_ratio <= 1.01 * tie_ratio_bounds[0]
        or state.tie_ratio >= 0.99 * tie_ratio_bounds[1]
        or state.radial_decay <= 1.01 * radial_decay_bounds[0]
        or state.radial_decay >= 0.99 * radial_decay_bounds[1]
    )
    state.condensate_endpoint = condensate_endpoint
    state.shape_boundary_limited = shape_boundary_limited
    state.boundary_limited = bool(
        condensate_endpoint or shape_boundary_limited
    )
    state.thermodynamic_valid = bool(
        state.minimum_gns_transfer_gap > float(gns_gap_tolerance)
    )
    state.success = bool(
        result.success
        and np.isfinite(state.energy_density)
        and state.thermodynamic_valid
    )
    messages = [str(result.message)]
    if condensate_endpoint:
        messages.append("the variational minimum is a condensate endpoint")
    if shape_boundary_limited:
        messages.append("fluctuation shape reached a parameter bound")
    if not state.thermodynamic_valid:
        messages.append("the dominant GNS transfer fixed point is not isolated")
    state.message = "; ".join(messages)
    state.optimized_parameters = {
        "condensate_density": state.condensate_density,
        "fluctuation_density": state.fluctuation_density,
        "condensate_fraction": state.condensate_fraction,
        "field_amplitude": state.field_amplitude,
        "field_phase": state.field_phase,
        "outer_gap": state.outer_gap,
        "tie_ratio": state.tie_ratio,
        "angular_memory_decay": state.angular_memory_decay,
        "radial_decay": state.radial_decay,
        "minimum_gns_transfer_gap": state.minimum_gns_transfer_gap,
    }
    return state


__all__ = [
    "DiluteBoseGas2D",
    "D2M1HierarchicalCLETTA2D",
    "D2M1NestedCLETTA2D",
    "GaussianPotentialBoseGas2D",
    "HierarchicalShellContraction",
    "RankOneDensityTransferChannel2D",
    "fixed_density_gns_nested_hletta_state",
    "fixed_density_nested_hletta_state",
    "optimize_condensate_gns_hletta_fixed_density",
    "optimize_condensate_nested_hletta_fixed_density",
    "optimize_nested_hletta_fixed_density",
]
