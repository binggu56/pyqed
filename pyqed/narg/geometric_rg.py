"""Hamiltonian geometric RG and covariant FRG for scalar field theory."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, gamma, pi

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi


def _sphere_area(dimension: int) -> float:
    return 2.0 * pi ** (0.5 * dimension) / gamma(0.5 * dimension)


@dataclass(frozen=True)
class Phi4GaussianCouplings:
    """Dimensionless couplings of ``j phi + r phi^2/2 + g phi^3/6 + u phi^4/24``."""

    source: float = 0.0
    mass2: float = 0.0
    cubic: float = 0.0
    quartic: float = 0.0

    def asarray(self) -> np.ndarray:
        return np.array(
            [self.source, self.mass2, self.cubic, self.quartic], dtype=float
        )

    @classmethod
    def from_array(cls, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (4,):
            raise ValueError("phi4 couplings must contain four values.")
        return cls(*values)


class Phi4GaussianShell:
    r"""Homogeneous Gaussian shell flow for continuum ``phi^4`` theory.

    The flow parameter is ``ell = log(Lambda_0 / Lambda)`` and all couplings
    and fields are dimensionless at the running cutoff.  Integrating the
    spatial momentum shell after its frequency has been integrated out gives

    ``d_ell u = D u - (D-2) phi u'/2 + A_d sqrt(1 + u'')``.
    """

    def __init__(self, spatial_dimension: int = 1):
        self.spatial_dimension = int(spatial_dimension)
        if self.spatial_dimension < 1:
            raise ValueError("spatial_dimension must be positive.")

    @property
    def spacetime_dimension(self) -> int:
        return self.spatial_dimension + 1

    @property
    def shell_density(self) -> float:
        """Radial shell measure per ``d ell`` at unit cutoff."""
        d = self.spatial_dimension
        return _sphere_area(d) / (2.0 * pi) ** d

    @property
    def fluctuation_prefactor(self) -> float:
        return 0.5 * self.shell_density

    @staticmethod
    def potential(field, couplings: Phi4GaussianCouplings):
        field = np.asarray(field)
        return (
            couplings.source * field
            + 0.5 * couplings.mass2 * field**2
            + couplings.cubic * field**3 / 6.0
            + couplings.quartic * field**4 / 24.0
        )

    @staticmethod
    def curvature(field, couplings: Phi4GaussianCouplings):
        field = np.asarray(field)
        return (
            couplings.mass2
            + couplings.cubic * field
            + 0.5 * couplings.quartic * field**2
        )

    @staticmethod
    def third_derivative(field, couplings: Phi4GaussianCouplings):
        return couplings.cubic + couplings.quartic * np.asarray(field)

    def beta_potential(
        self,
        field,
        couplings: Phi4GaussianCouplings,
        *,
        inertia=1.0,
        stiffness=1.0,
    ):
        field = np.asarray(field, dtype=float)
        curvature = self.curvature(field, couplings)
        inertia = np.asarray(inertia, dtype=float)
        stiffness = np.asarray(stiffness, dtype=float)
        if np.any(stiffness + curvature <= 0.0) or np.any(inertia <= 0.0):
            raise ValueError("the Gaussian shell frequency is not real.")
        dimension = self.spacetime_dimension
        field_dimension = 0.5 * (dimension - 2.0)
        derivative = (
            couplings.source
            + couplings.mass2 * field
            + 0.5 * couplings.cubic * field**2
            + couplings.quartic * field**3 / 6.0
        )
        return (
            dimension * self.potential(field, couplings)
            - field_dimension * field * derivative
            + self.fluctuation_prefactor
            * np.sqrt((stiffness + curvature) / inertia)
        )

    def beta(self, couplings: Phi4GaussianCouplings) -> Phi4GaussianCouplings:
        """Return the polynomial flow obtained from the Gaussian shell determinant."""
        r = couplings.mass2
        g = couplings.cubic
        u = couplings.quartic
        w = 1.0 + r
        if w <= 0.0:
            raise ValueError("mass2 must be greater than -1 at the running cutoff.")

        a = self.fluctuation_prefactor
        dimension = self.spacetime_dimension
        field_dimension = 0.5 * (dimension - 2.0)
        source = (dimension - field_dimension) * couplings.source
        source += a * g / (2.0 * np.sqrt(w))
        mass2 = 2.0 * r + a * (
            u / (2.0 * np.sqrt(w)) - g * g / (4.0 * w**1.5)
        )
        cubic = (dimension - 3.0 * field_dimension) * g
        cubic += a * (
            3.0 * g**3 / (8.0 * w**2.5)
            - 3.0 * g * u / (4.0 * w**1.5)
        )
        quartic = (dimension - 4.0 * field_dimension) * u
        quartic += a * (
            -15.0 * g**4 / (16.0 * w**3.5)
            + 9.0 * g * g * u / (4.0 * w**2.5)
            - 3.0 * u * u / (4.0 * w**1.5)
        )
        return Phi4GaussianCouplings(source, mass2, cubic, quartic)

    def metric_rate(
        self,
        field,
        couplings: Phi4GaussianCouplings,
        *,
        inertia=1.0,
        inertia_derivative=0.0,
        stiffness=1.0,
        stiffness_derivative=0.0,
    ):
        """Quantum-metric density contributed by one infinitesimal shell."""
        field = np.asarray(field, dtype=float)
        spring = np.asarray(stiffness, dtype=float) + self.curvature(
            field, couplings
        )
        inertia = np.asarray(inertia, dtype=float)
        inertia_derivative = np.asarray(inertia_derivative, dtype=float)
        stiffness_derivative = np.asarray(stiffness_derivative, dtype=float)
        if np.any(spring <= 0.0) or np.any(inertia <= 0.0):
            raise ValueError("the Gaussian shell frequency is not real.")
        derivative = self.third_derivative(field, couplings)
        log_kernel_derivative = 0.5 * (
            inertia_derivative / inertia
            + (stiffness_derivative + derivative) / spring
        )
        return self.shell_density * log_kernel_derivative**2 / 8.0

    def weighted_metric_rate(
        self,
        field,
        couplings: Phi4GaussianCouplings,
        *,
        energy: float = 0.0,
        inertia=1.0,
        inertia_derivative=0.0,
        stiffness=1.0,
        stiffness_derivative=0.0,
    ):
        """Two-boson resolvent-weighted metric density for one shell."""
        field = np.asarray(field, dtype=float)
        spring = np.asarray(stiffness, dtype=float) + self.curvature(
            field, couplings
        )
        inertia = np.asarray(inertia, dtype=float)
        if np.any(spring <= 0.0) or np.any(inertia <= 0.0):
            raise ValueError("the Gaussian shell frequency is not real.")
        denominator = float(energy) - 2.0 * np.sqrt(spring / inertia)
        if np.any(np.isclose(denominator, 0.0)):
            raise ValueError("energy lies on the Gaussian two-boson shell.")
        return self.metric_rate(
            field,
            couplings,
            inertia=inertia,
            inertia_derivative=inertia_derivative,
            stiffness=stiffness,
            stiffness_derivative=stiffness_derivative,
        ) / denominator

    def inertia_beta(
        self,
        field,
        couplings: Phi4GaussianCouplings,
        *,
        inertia,
        inertia_derivative,
        stiffness=1.0,
        stiffness_derivative=0.0,
        energy: float = 0.0,
    ):
        r"""Flow of ``Z`` in ``Pi^2/(2 Z)`` from the weighted metric."""
        field = np.asarray(field, dtype=float)
        inertia = np.asarray(inertia, dtype=float)
        inertia_derivative = np.asarray(inertia_derivative, dtype=float)
        field_dimension = 0.5 * (self.spacetime_dimension - 2.0)
        weighted = self.weighted_metric_rate(
            field,
            couplings,
            energy=energy,
            inertia=inertia,
            inertia_derivative=inertia_derivative,
            stiffness=stiffness,
            stiffness_derivative=stiffness_derivative,
        )
        return -field_dimension * field * inertia_derivative - 2.0 * weighted

    def beta_z2(self, couplings: Phi4GaussianCouplings, inertia2: float = 0.0):
        r"""Closed ``Z2`` flow for ``Z(phi) = 1 + z2 phi^2 / 2``.

        The source and cubic coupling must vanish.  The returned coupling flow
        includes the feedback of ``z2`` into the Gaussian shell determinant.
        """
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("beta_z2 requires a Z2-symmetric potential.")
        r = couplings.mass2
        u = couplings.quartic
        z2 = float(inertia2)
        w = 1.0 + r
        if w <= 0.0:
            raise ValueError("mass2 must be greater than -1 at the running cutoff.")
        a = self.fluctuation_prefactor
        density = self.shell_density
        dimension = self.spacetime_dimension
        delta = u - w * z2
        mass2 = 2.0 * r + a * delta / (2.0 * np.sqrt(w))
        quartic = (4.0 - dimension) * u + a * (
            -3.0 * z2 * delta / np.sqrt(w)
            - 3.0 * delta * delta / (4.0 * w**1.5)
        )
        inertia2_beta = -(dimension - 2.0) * z2
        inertia2_beta += density * (z2 + u / w) ** 2 / (16.0 * np.sqrt(w))
        return (
            Phi4GaussianCouplings(mass2=mass2, quartic=quartic),
            float(inertia2_beta),
        )

    def external_kinetic_rates(
        self,
        field,
        couplings: Phi4GaussianCouplings,
        *,
        momentum_step: float | None = None,
    ):
        r"""Project the shell self-energy onto frequency and momentum squared.

        This is the direct two-point bubble projection for the current sharp
        spatial shell.  A finite ``momentum_step`` evaluates both derivatives
        by centered differences and is currently available in one spatial
        dimension; the default returns their analytic zero-step limits.
        """
        field = float(field)
        spring = 1.0 + float(self.curvature(field, couplings))
        if spring <= 0.0:
            raise ValueError("the Gaussian shell frequency is not real.")
        vertex2 = float(self.third_derivative(field, couplings)) ** 2
        frequency = np.sqrt(spring)
        density = self.shell_density
        temporal = density * vertex2 / (32.0 * frequency**5)
        spatial = density * vertex2 / (32.0 * frequency**7)
        spatial *= 3.0 * frequency**2 - 10.0 / self.spatial_dimension
        if momentum_step is None:
            return float(temporal), float(spatial)

        if self.spatial_dimension != 1:
            raise NotImplementedError(
                "finite-step external projection is currently implemented only in 1D."
            )
        step = float(momentum_step)
        if step <= 0.0:
            raise ValueError("momentum_step must be positive.")

        def temporal_bubble(external_frequency):
            return 1.0 / (
                frequency * (external_frequency**2 + 4.0 * frequency**2)
            )

        def spatial_bubble(external_momentum):
            total = 0.0
            for shell_momentum in (-1.0, 1.0):
                shifted2 = (
                    shell_momentum + external_momentum
                ) ** 2 + spring - 1.0
                if shifted2 <= 0.0:
                    raise ValueError("the shifted shell propagator is unstable.")
                shifted = np.sqrt(shifted2)
                total += 1.0 / (
                    2.0
                    * frequency
                    * shifted
                    * (frequency + shifted)
                )
            return 0.5 * total

        prefactor = -0.5 * density * vertex2

        def coefficient(function):
            return prefactor * (
                function(step) - 2.0 * function(0.0) + function(-step)
            ) / (2.0 * step**2)

        return float(coefficient(temporal_bubble)), float(
            coefficient(spatial_bubble)
        )

    def level2_response(
        self,
        field,
        couplings: Phi4GaussianCouplings,
        momentum,
        *,
        angular_order: int = 64,
    ):
        r"""Return the nonuniform Gaussian QGRG response of one shell.

        ``momentum`` is the dimensionless external spatial momentum in units
        of the running cutoff.  The response is evaluated for a unit shell
        momentum and angular-averaged in ``spatial_dimension`` dimensions.
        It contains the linear Gaussian-kernel response, the static two-point
        bubble, and the normalized-vacuum overlap metric.  The shell density
        is included, so the returned quantities are rates per ``d ell``.

        This is the bulk sharp-shell response.  It deliberately does not
        differentiate the sharp cutoff boundary with respect to external
        momentum; use it either at small momentum or with a smooth shell
        implementation when boundary-sensitive derivative coefficients are
        required.
        """
        field = float(field)
        momentum = np.atleast_1d(np.asarray(momentum, dtype=float))
        if momentum.ndim != 1 or not np.all(np.isfinite(momentum)):
            raise ValueError("momentum must be a finite scalar or one-dimensional array.")
        angular_order = int(angular_order)
        if angular_order < 2:
            raise ValueError("angular_order must be at least 2.")

        spring = 1.0 + float(self.curvature(field, couplings))
        if spring <= 0.0:
            raise ValueError("the Gaussian shell frequency is not real.")
        frequency = np.sqrt(spring)
        mass_response = float(self.third_derivative(field, couplings))

        if self.spatial_dimension == 1:
            cosine = np.array([-1.0, 1.0])
            angular_weights = np.array([0.5, 0.5])
        else:
            exponent = 0.5 * (self.spatial_dimension - 3.0)
            cosine, angular_weights = roots_jacobi(
                angular_order, exponent, exponent
            )
            angular_weights = angular_weights / np.sum(angular_weights)

        q = momentum[:, None]
        shifted_spring = spring + q * q + 2.0 * q * cosine[None, :]
        if np.any(shifted_spring <= 0.0):
            raise ValueError("an external momentum reaches an unstable shifted mode.")
        shifted_frequency = np.sqrt(shifted_spring)
        average = lambda values: np.sum(values * angular_weights[None, :], axis=1)

        kernel_response = mass_response * average(
            1.0 / (frequency + shifted_frequency)
        )
        bubble = self.shell_density * average(
            1.0
            / (
                2.0
                * frequency
                * shifted_frequency
                * (frequency + shifted_frequency)
            )
        )
        two_point_rate = -0.5 * mass_response**2 * bubble
        overlap_metric = self.shell_density * mass_response**2 * average(
            1.0
            / (
                frequency
                * shifted_frequency
                * (frequency + shifted_frequency) ** 2
            )
        ) / 8.0
        temporal_rate, spatial_rate = self.external_kinetic_rates(
            field, couplings
        )
        return {
            "momentum": momentum.copy(),
            "frequency": float(frequency),
            "mass_response": float(mass_response),
            "kernel_response": kernel_response,
            "bubble": bubble,
            "two_point_rate": two_point_rate,
            "overlap_metric": overlap_metric,
            "temporal_rate": float(temporal_rate),
            "spatial_rate": float(spatial_rate),
        }

    def anomalous_dimension(
        self,
        couplings: Phi4GaussianCouplings,
        *,
        projection: str = "matched",
    ) -> float:
        r"""Return the minimum-normalized geometric anomalous dimension.

        The shell correction is evaluated at the broken-phase minimum.  The
        default matches the temporal metric to ``Z_x``; ``projection='spatial'``
        uses the sharp-shell external-momentum derivative directly.
        """
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("anomalous_dimension requires a Z2-symmetric potential.")
        if couplings.quartic < 0.0:
            raise ValueError("quartic must be nonnegative.")
        if couplings.mass2 >= 0.0 or couplings.quartic == 0.0:
            return 0.0
        minimum = np.sqrt(
            -6.0 * couplings.mass2 / couplings.quartic
        )
        temporal, spatial = self.external_kinetic_rates(minimum, couplings)
        if projection == "matched":
            return temporal
        if projection == "spatial":
            return spatial
        raise ValueError("projection must be 'matched' or 'spatial'.")

    def beta_lpa_prime(
        self,
        couplings: Phi4GaussianCouplings,
        *,
        projection: str = "matched",
    ):
        r"""LPA-prime flow using a selected minimum-normalized projection."""
        eta = self.anomalous_dimension(couplings, projection=projection)
        beta = self.beta(couplings)
        return (
            Phi4GaussianCouplings(
                source=beta.source,
                mass2=beta.mass2 - eta * couplings.mass2,
                cubic=beta.cubic,
                quartic=beta.quartic - 2.0 * eta * couplings.quartic,
            ),
            eta,
        )

    @staticmethod
    def _shell_quadrature(log_width: float, order: int, cutoff: float):
        log_width = float(log_width)
        cutoff = float(cutoff)
        order = int(order)
        if log_width <= 0.0 or cutoff <= 0.0 or order < 2:
            raise ValueError("log_width and cutoff must be positive and order >= 2.")
        lower = cutoff * np.exp(-log_width)
        nodes, weights = leggauss(order)
        positive = 0.5 * (cutoff - lower) * nodes + 0.5 * (cutoff + lower)
        positive_weights = 0.5 * (cutoff - lower) * weights
        points = np.concatenate([-positive[::-1], positive])
        full_weights = np.concatenate([positive_weights[::-1], positive_weights])
        return points, full_weights, lower, cutoff

    def residual_corrections(
        self,
        field: float,
        couplings: Phi4GaussianCouplings,
        *,
        log_width: float,
        energy: float = 0.0,
        quadrature_order: int = 80,
        cutoff: float = 1.0,
    ):
        """Return finite-shell three/four-boson Feshbach energy densities.

        This direct quadrature currently covers one spatial dimension.  It is
        intended as a finite-shell diagnostic; both terms vanish faster than
        ``d ell`` for an infinitesimal shell.
        """
        if self.spatial_dimension != 1:
            raise NotImplementedError(
                "finite residual quadrature is currently implemented only in 1D."
            )
        k, weights, lower, upper = self._shell_quadrature(
            log_width, quadrature_order, cutoff
        )
        mass2 = float(self.curvature(float(field), couplings))
        if lower * lower + mass2 <= 0.0:
            raise ValueError("the finite shell contains an unstable Gaussian mode.")

        def inside(values):
            absolute = np.abs(values)
            return (absolute >= lower) & (absolute <= upper)

        def omega(values):
            return np.sqrt(values * values + mass2)

        k1 = k[:, None]
        k2 = k[None, :]
        k3 = -(k1 + k2)
        mask3 = inside(k3)
        omega1 = omega(k1)
        omega2 = omega(k2)
        omega3 = omega(k3)
        weight12 = weights[:, None] * weights[None, :] / (2.0 * pi) ** 2
        denominator3 = energy - omega1 - omega2 - omega3
        integrand3 = weight12 / (8.0 * omega1 * omega2 * omega3) / denominator3
        cubic = float(self.third_derivative(float(field), couplings))
        correction3 = cubic * cubic * np.sum(integrand3[mask3]) / 6.0

        correction4 = 0.0
        for index, first in enumerate(k):
            second = k[:, None]
            third = k[None, :]
            fourth = -(first + second + third)
            mask4 = inside(fourth)
            omega_first = omega(first)
            omega_second = omega(second)
            omega_third = omega(third)
            omega_fourth = omega(fourth)
            combined_weight = (
                weights[index] * weights[:, None] * weights[None, :] / (2.0 * pi) ** 3
            )
            denominator4 = (
                energy - omega_first - omega_second - omega_third - omega_fourth
            )
            integrand4 = combined_weight / (
                16.0 * omega_first * omega_second * omega_third * omega_fourth
            ) / denominator4
            correction4 += np.sum(integrand4[mask4])
        correction4 *= couplings.quartic**2 / 24.0
        return {
            "three_boson": float(correction3),
            "four_boson": float(correction4),
            "total": float(correction3 + correction4),
        }


def _local_derivative_matrix(grid, order: int, stencil_size: int) -> np.ndarray:
    """Build a local-polynomial differentiation matrix on a one-dimensional grid."""
    grid = np.asarray(grid, dtype=float)
    order = int(order)
    width = min(int(stencil_size), grid.size)
    if width <= order:
        raise ValueError("stencil_size must exceed the derivative order.")
    matrix = np.zeros((grid.size, grid.size), dtype=float)
    half = width // 2
    for index in range(grid.size):
        start = min(max(index - half, 0), grid.size - width)
        selected = np.arange(start, start + width)
        offsets = grid[selected] - grid[index]
        scale = np.max(np.abs(offsets))
        scaled = offsets / scale
        powers = np.arange(width)[:, None]
        vandermonde = scaled[None, :] ** powers
        target = np.zeros(width)
        target[order] = factorial(order) / scale**order
        matrix[index, selected] = np.linalg.solve(vandermonde, target)
    return matrix


class Phi4FunctionalQGRG:
    r"""Functional Gaussian-frame QGRG on a background-field grid.

    This is the Hamiltonian spatial-shell construction.  Its local shell
    Hamiltonian at each background is

    ``Pi^2 / (2 Z_t) + (Z_x + U'') varphi^2 / 2``.

    The zero-point energy evolves ``U``.  The resolvent-weighted frame metric
    evolves ``Z_t``, while the spatial external-momentum response evolves
    ``Z_x``.  No Lorentz matching is imposed on the two kinetic functions.
    """

    def __init__(
        self,
        field,
        *,
        spatial_dimension: int = 1,
        stencil_size: int = 7,
    ):
        self.field = np.asarray(field, dtype=float)
        if self.field.ndim != 1 or self.field.size < 5:
            raise ValueError("field must be a one-dimensional grid with at least 5 points.")
        if np.any(np.diff(self.field) <= 0.0):
            raise ValueError("field points must be strictly increasing.")
        self.spatial_dimension = int(spatial_dimension)
        if self.spatial_dimension < 1:
            raise ValueError("spatial_dimension must be positive.")
        self.stencil_size = min(int(stencil_size), self.field.size)
        if self.stencil_size < 5:
            raise ValueError("stencil_size must be at least 5.")
        self._derivatives = tuple(
            _local_derivative_matrix(self.field, order, self.stencil_size)
            for order in (1, 2, 3, 4)
        )

    @property
    def spacetime_dimension(self) -> int:
        return self.spatial_dimension + 1

    @property
    def shell_density(self) -> float:
        d = self.spatial_dimension
        return _sphere_area(d) / (2.0 * pi) ** d

    @property
    def fluctuation_prefactor(self) -> float:
        return 0.5 * self.shell_density

    def derivative(self, values, order: int) -> np.ndarray:
        values = self._field_values(values, "values")
        if order not in (1, 2, 3, 4):
            raise ValueError("order must be 1, 2, 3, or 4.")
        return self._derivatives[order - 1] @ values

    def _field_values(self, values, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != self.field.shape:
            raise ValueError(f"{name} must have shape {self.field.shape}.")
        return values

    def shell_geometry(self, potential, *, inertia=None, stiffness=None):
        r"""Return the local frequency, frame metric, and kinetic shell rates."""
        potential = self._field_values(potential, "potential")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self._field_values(inertia, "inertia")
        stiffness = self._field_values(stiffness, "stiffness")
        if np.any(inertia <= 0.0):
            raise ValueError("inertia must be positive.")

        curvature = self.derivative(potential, 2)
        third = self.derivative(potential, 3)
        inertia_derivative = self.derivative(inertia, 1)
        stiffness_derivative = self.derivative(stiffness, 1)
        spring = stiffness + curvature
        if np.any(spring <= 0.0):
            raise ValueError("the Gaussian shell frequency is not real.")

        frequency = np.sqrt(spring / inertia)
        log_kernel_derivative = 0.5 * (
            inertia_derivative / inertia
            + (stiffness_derivative + third) / spring
        )
        metric_rate = self.shell_density * log_kernel_derivative**2 / 8.0
        temporal_rate = metric_rate / frequency
        spatial_rate = temporal_rate * stiffness * (
            3.0 - 10.0 * stiffness / (self.spatial_dimension * spring)
        )
        return {
            "curvature": curvature,
            "third_derivative": third,
            "spring": spring,
            "frequency": frequency,
            "metric_rate": metric_rate,
            "temporal_rate": temporal_rate,
            "spatial_rate": spatial_rate,
        }

    def rates(
        self,
        potential,
        *,
        inertia=None,
        stiffness=None,
        energy: float = 0.0,
    ):
        r"""Return ``(d_ell U, d_ell Z_t, d_ell Z_x)`` on the field grid.

        ``energy=0`` gives the adiabatic Feshbach resolvent used by the
        infinitesimal shell flow.  Other energies retain its explicit
        state-dependent denominator in the temporal response.
        """
        potential = self._field_values(potential, "potential")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self._field_values(inertia, "inertia")
        stiffness = self._field_values(stiffness, "stiffness")
        geometry = self.shell_geometry(
            potential, inertia=inertia, stiffness=stiffness
        )
        derivative = self.derivative(potential, 1)
        field_dimension = 0.5 * (self.spacetime_dimension - 2.0)
        potential_rate = (
            self.spacetime_dimension * potential
            - field_dimension * self.field * derivative
            + self.fluctuation_prefactor * geometry["frequency"]
        )

        inertia_derivative = self.derivative(inertia, 1)
        stiffness_derivative = self.derivative(stiffness, 1)
        denominator = float(energy) - 2.0 * geometry["frequency"]
        if np.any(np.isclose(denominator, 0.0)):
            raise ValueError("energy lies on the Gaussian two-boson shell.")
        temporal_rate = -2.0 * geometry["metric_rate"] / denominator
        spatial_ratio = stiffness * (
            3.0
            - 10.0
            * stiffness
            / (self.spatial_dimension * geometry["spring"])
        )
        inertia_rate = (
            -field_dimension * self.field * inertia_derivative + temporal_rate
        )
        stiffness_rate = (
            -field_dimension * self.field * stiffness_derivative
            + temporal_rate * spatial_ratio
        )
        return potential_rate, inertia_rate, stiffness_rate


class Phi4WegnerHoughtonLPA:
    r"""Global sharp-cutoff Wegner--Houghton fixed point in the LPA.

    The vacuum-energy ambiguity is removed by solving for the force
    ``f(phi) = U'(phi)``.  For RG time increasing toward the infrared,

    ``beta_f = a_D f''/(1 + f') + (D-delta) f - delta phi f'``,

    where ``delta=(D-2+eta)/2``.  The interacting solution is selected by
    ``f(0)=0`` and the large-field condition
    ``phi f'=(D/delta-1) f``.  Field-extent continuation is essential because
    the sharp-cutoff shooting problem is exponentially ill-conditioned.
    """

    def __init__(self, spacetime_dimension=3, *, eta=0.0):
        self.spacetime_dimension = float(spacetime_dimension)
        self.eta = float(eta)
        if self.spacetime_dimension <= 2.0:
            if self.spacetime_dimension < 2.0:
                raise ValueError("spacetime_dimension must be at least two.")
        self.field_dimension = 0.5 * (
            self.spacetime_dimension - 2.0 + self.eta
        )
        dimension = self.spacetime_dimension
        self.loop_prefactor = (
            0.5
            * 2.0
            * pi ** (0.5 * dimension)
            / gamma(0.5 * dimension)
            / (2.0 * pi) ** dimension
        )
        self.fixed_point_history = None
        self.fixed_solution = None
        self.fixed_curvature = None
        self.stability_eigenvalues = None
        self.correlation_exponent = None

    @property
    def large_field_power(self):
        if self.field_dimension <= 0.0:
            return float("inf")
        return self.spacetime_dimension / self.field_dimension

    def force_beta(self, field, force, force_prime, force_second):
        """Evaluate the differentiated Wegner--Houghton flow."""
        field = np.asarray(field, dtype=float)
        force = np.asarray(force, dtype=float)
        force_prime = np.asarray(force_prime, dtype=float)
        force_second = np.asarray(force_second, dtype=float)
        if np.any(1.0 + force_prime <= 0.0):
            raise ValueError("the sharp-cutoff inverse propagator is singular.")
        return (
            self.loop_prefactor * force_second / (1.0 + force_prime)
            + (self.spacetime_dimension - self.field_dimension) * force
            - self.field_dimension * field * force_prime
        )

    def _fixed_ode(self, field, values):
        force, force_prime = values
        force_second = (
            (1.0 + force_prime)
            * (
                self.field_dimension * field * force_prime
                - (self.spacetime_dimension - self.field_dimension) * force
            )
            / self.loop_prefactor
        )
        return np.vstack((force_prime, force_second))

    def solve_fixed_point(
        self,
        *,
        field_maxima=(0.5, 0.6, 0.8, 1.0),
        initial_curvature=-0.46,
        mesh_points=151,
        tolerance=1.0e-7,
        max_nodes=50000,
    ):
        """Continue the global interacting solution through field extent."""
        from scipy.integrate import solve_bvp

        if self.field_dimension <= 0.0:
            raise ValueError(
                "the LPA field dimension is zero, so no isolated power-law "
                "large-field boundary condition exists; supply eta>0 or use "
                "a kinetic truncation"
            )
        field_maxima = np.asarray(field_maxima, dtype=float)
        if (
            field_maxima.ndim != 1
            or field_maxima.size == 0
            or np.any(field_maxima <= 0.0)
            or np.any(np.diff(field_maxima) <= 0.0)
        ):
            raise ValueError("field_maxima must be strictly increasing and positive.")
        mesh_points = int(mesh_points)
        if mesh_points < 25:
            raise ValueError("mesh_points must be at least 25.")
        force_power = self.large_field_power - 1.0
        history = []
        previous = None

        for field_max in field_maxima:
            field = np.linspace(0.0, field_max, mesh_points)
            if previous is None:
                curvature = float(initial_curvature)
                cubic_coefficient = (
                    (1.0 + curvature)
                    * curvature
                    * (2.0 * self.field_dimension - self.spacetime_dimension)
                    / (6.0 * self.loop_prefactor)
                )
                seed_degree = max(7, int(np.ceil(force_power)) + 2)
                if seed_degree % 2 == 0:
                    seed_degree += 1
                denominator = (
                    (seed_degree - force_power) * field_max**seed_degree
                )
                numerator = (
                    curvature * (1.0 - force_power) * field_max
                    + cubic_coefficient
                    * (3.0 - force_power)
                    * field_max**3
                )
                tail = -numerator / denominator
                force = (
                    curvature * field
                    + cubic_coefficient * field**3
                    + tail * field**seed_degree
                )
                force_prime = (
                    curvature
                    + 3.0 * cubic_coefficient * field**2
                    + seed_degree * tail * field ** (seed_degree - 1)
                )
                guess = np.vstack((force, force_prime))
            else:
                old_max = float(previous.x[-1])
                guess = np.empty((2, field.size))
                inside = field <= old_max
                guess[:, inside] = previous.sol(field[inside])
                old_force = float(previous.sol(old_max)[0])
                amplitude = old_force / old_max**force_power
                guess[0, ~inside] = amplitude * field[~inside] ** force_power
                guess[1, ~inside] = (
                    force_power
                    * amplitude
                    * field[~inside] ** (force_power - 1.0)
                )

            def boundary(left, right, maximum=field_max):
                return np.array(
                    [
                        left[0],
                        maximum * right[1] - force_power * right[0],
                    ]
                )

            solution = solve_bvp(
                self._fixed_ode,
                boundary,
                field,
                guess,
                tol=tolerance,
                max_nodes=int(max_nodes),
            )
            residual = float(np.max(solution.rms_residuals))
            interacting = bool(abs(solution.y[1, 0]) > 1.0e-4)
            converged = bool(solution.success and interacting)
            history.append(
                {
                    "field_max": float(field_max),
                    "curvature": float(solution.y[1, 0]),
                    "nodes": int(solution.x.size),
                    "max_residual": residual,
                    "success": converged,
                }
            )
            if not converged:
                self.success = False
                self.message = (
                    "global fixed-point continuation failed at "
                    f"field_max={field_max:g}: {solution.message}"
                )
                self.fixed_point_history = history
                self.fixed_solution = solution
                return self
            previous = solution

        self.fixed_point_history = history
        self.fixed_solution = previous
        self.fixed_curvature = float(previous.y[1, 0])
        self.field_max = float(field_maxima[-1])
        self.success = True
        self.message = "global Wegner--Houghton fixed point converged"
        return self

    @staticmethod
    def _chebyshev_matrices(field_max, points):
        points = int(points)
        index = np.arange(points)
        field = 0.5 * field_max * (
            1.0 - np.cos(pi * index / (points - 1))
        )
        barycentric = (-1.0) ** index
        barycentric[[0, -1]] *= 0.5
        difference = field[:, None] - field[None, :]
        first = (
            barycentric[None, :] / barycentric[:, None]
        ) / (difference + np.eye(points))
        np.fill_diagonal(first, 0.0)
        np.fill_diagonal(first, -np.sum(first, axis=1))
        return field, first, first @ first

    def stability_spectrum(
        self,
        *,
        points=60,
        eigenvalue_window=(-20.0, 5.0),
        imaginary_tolerance=1.0e-5,
    ):
        """Solve the global even-potential stability eigenproblem."""
        if self.fixed_solution is None or not self.success:
            raise ValueError("solve the global fixed point before stability analysis.")
        field, first, second = self._chebyshev_matrices(
            self.field_max, points
        )
        force, force_prime = self.fixed_solution.sol(field)
        force_second = self._fixed_ode(
            field, np.vstack((force, force_prime))
        )[1]
        identity = np.eye(field.size)
        operator = self.loop_prefactor * (
            np.diag(1.0 / (1.0 + force_prime)) @ second
            - np.diag(force_second / (1.0 + force_prime) ** 2) @ first
        )
        operator += (
            self.spacetime_dimension - self.field_dimension
        ) * identity
        operator -= self.field_dimension * np.diag(field) @ first

        reduced = np.empty((field.size - 1, field.size - 1))
        reduced[:-1] = operator[1:-1, 1:]
        reduced[-1] = (
            (self.spacetime_dimension - self.field_dimension)
            * identity[-1, 1:]
            - self.field_dimension * self.field_max * first[-1, 1:]
        )
        eigenvalues, eigenvectors = np.linalg.eig(reduced)
        order = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        lower, upper = map(float, eigenvalue_window)
        physical = (
            (np.abs(eigenvalues.imag) <= imaginary_tolerance)
            & (eigenvalues.real >= lower)
            & (eigenvalues.real <= upper)
        )
        self.stability_matrix = reduced
        self.stability_all_eigenvalues = eigenvalues
        self.stability_eigenvalues = eigenvalues[physical]
        self.stability_eigenvectors = eigenvectors[:, physical]
        relevant = self.stability_eigenvalues[
            self.stability_eigenvalues.real > 1.0e-7
        ]
        self.relevant_eigenvalue = (
            float(relevant[0].real) if relevant.size else float("nan")
        )
        self.correlation_exponent = 1.0 / self.relevant_eigenvalue
        self.stability_field = field
        return self.stability_eigenvalues

    def potential(self, field):
        """Return the fixed potential with the gauge choice ``U(0)=0``."""
        from scipy.integrate import cumulative_trapezoid

        if self.fixed_solution is None or not self.success:
            raise ValueError("solve the global fixed point before evaluating U.")
        field = np.asarray(field, dtype=float)
        if field.ndim != 1 or np.any(np.diff(field) < 0.0):
            raise ValueError("field must be an increasing one-dimensional grid.")
        force = self.fixed_solution.sol(field)[0]
        return cumulative_trapezoid(force, field, initial=0.0)


class Phi4VariationalQGRG(Phi4FunctionalQGRG):
    r"""Finite-shell QGRG with a variational Gaussian frame and Feshbach residual.

    The quartic Gaussian expectation is optimized self-consistently.  Cubic
    and normal-ordered quartic shell interactions are retained in the
    three- and four-boson Feshbach correction.  The implementation currently
    covers one spatial dimension.
    """

    def __init__(
        self,
        field,
        *,
        log_width: float = 0.2,
        quadrature_order: int = 40,
        cutoff: float = 1.0,
        stencil_size: int = 7,
    ):
        super().__init__(
            field,
            spatial_dimension=1,
            stencil_size=stencil_size,
        )
        self.log_width = float(log_width)
        self.quadrature_order = int(quadrature_order)
        self.cutoff = float(cutoff)
        self._momenta, weights, self.lower_cutoff, self.upper_cutoff = (
            Phi4GaussianShell._shell_quadrature(
                self.log_width, self.quadrature_order, self.cutoff
            )
        )
        self._momentum_weights = weights / (2.0 * pi)
        self.frame = None
        self.feshbach = None

    @staticmethod
    def _gap_solution(base, quartic, inertia, weights):
        base = np.asarray(base, dtype=float)
        quartic = float(quartic)
        inertia = float(inertia)
        if inertia <= 0.0:
            raise ValueError("inertia must be positive.")
        if quartic < -1.0e-10:
            raise ValueError("the variational shell requires nonnegative U''''.")
        quartic = max(quartic, 0.0)

        def variance(hartree):
            spring = base + hartree
            if np.any(spring <= 0.0):
                return np.inf
            frequency = np.sqrt(spring / inertia)
            return float(np.sum(weights / (2.0 * inertia * frequency)))

        if quartic == 0.0:
            if np.any(base <= 0.0):
                raise ValueError("the Gaussian shell frequency is not real.")
            return 0.0, variance(0.0)

        lower = max(0.0, -float(np.min(base)) + 1.0e-13)

        def residual(hartree):
            return hartree - 0.5 * quartic * variance(hartree)

        lower_residual = residual(lower)
        if lower_residual > 0.0:
            raise ValueError("no stable variational Hartree solution was found.")
        upper = max(1.0, lower + 1.0, quartic)
        for _ in range(80):
            if residual(upper) > 0.0:
                break
            upper *= 2.0
        else:
            raise ValueError("the variational Hartree gap did not bracket a root.")
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            if residual(midpoint) > 0.0:
                upper = midpoint
            else:
                lower = midpoint
        hartree = 0.5 * (lower + upper)
        return hartree, variance(hartree)

    def variational_frame(self, potential, *, inertia=None, stiffness=None):
        r"""Optimize the translation-invariant Gaussian shell covariance."""
        potential = self._field_values(potential, "potential")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self._field_values(inertia, "inertia")
        stiffness = self._field_values(stiffness, "stiffness")
        curvature = self.derivative(potential, 2)
        cubic = self.derivative(potential, 3)
        quartic = self.derivative(potential, 4)
        quartic[np.abs(quartic) < 1.0e-9] = 0.0

        nfield = self.field.size
        nmomentum = self._momenta.size
        frequency = np.empty((nfield, nmomentum))
        kernel = np.empty_like(frequency)
        variance = np.empty(nfield)
        hartree = np.empty(nfield)
        energy = np.empty(nfield)
        momentum2 = self._momenta**2
        for index in range(nfield):
            base = stiffness[index] * momentum2 + curvature[index]
            hartree[index], variance[index] = self._gap_solution(
                base,
                quartic[index],
                inertia[index],
                self._momentum_weights,
            )
            frequency[index] = np.sqrt(
                (base + hartree[index]) / inertia[index]
            )
            kernel[index] = inertia[index] * frequency[index]
            quadratic_energy = np.sum(
                self._momentum_weights
                * (
                    0.25 * frequency[index]
                    + base / (4.0 * inertia[index] * frequency[index])
                )
            )
            energy[index] = (
                quadratic_energy
                + 0.125 * quartic[index] * variance[index] ** 2
            )

        log_kernel_derivative = self._derivatives[0] @ np.log(kernel)
        metric_integrand = log_kernel_derivative**2 / 8.0
        metric = metric_integrand @ self._momentum_weights
        frame = {
            "curvature": curvature,
            "cubic": cubic,
            "quartic": quartic,
            "hartree": hartree,
            "variance": variance,
            "frequency": frequency,
            "kernel": kernel,
            "log_kernel_derivative": log_kernel_derivative,
            "metric_integrand": metric_integrand,
            "metric": metric,
            "energy": energy,
        }
        self.frame = frame
        return frame

    def feshbach_energy(self, frame, *, inertia, stiffness, energy: float = 0.0):
        r"""Evaluate dressed three- and four-boson residual shell energies."""
        inertia = self._field_values(inertia, "inertia")
        stiffness = self._field_values(stiffness, "stiffness")
        points = self._momenta
        weights = self._momentum_weights
        lower = self.lower_cutoff
        upper = self.upper_cutoff

        def inside(values):
            absolute = np.abs(values)
            return (absolute >= lower) & (absolute <= upper)

        three = np.zeros(self.field.size)
        four = np.zeros(self.field.size)
        k1 = points[:, None]
        k2 = points[None, :]
        k3 = -(k1 + k2)
        mask3 = inside(k3)
        weight12 = weights[:, None] * weights[None, :]
        for field_index in range(self.field.size):
            zt = inertia[field_index]
            zx = stiffness[field_index]
            mass = frame["curvature"][field_index] + frame["hartree"][field_index]

            def omega(momentum):
                return np.sqrt((zx * momentum**2 + mass) / zt)

            omega1 = omega(k1)
            omega2 = omega(k2)
            omega3 = np.ones_like(k3)
            omega3[mask3] = omega(k3[mask3])
            denominator3 = float(energy) - omega1 - omega2 - omega3
            integrand3 = weight12 / (
                8.0 * zt**3 * omega1 * omega2 * omega3 * denominator3
            )
            three[field_index] = (
                frame["cubic"][field_index] ** 2
                * np.sum(integrand3[mask3])
                / 6.0
            )

            correction4 = 0.0
            for index, first in enumerate(points):
                second = points[:, None]
                third = points[None, :]
                fourth = -(first + second + third)
                mask4 = inside(fourth)
                omega_first = omega(first)
                omega_second = omega(second)
                omega_third = omega(third)
                omega_fourth = np.ones_like(fourth)
                omega_fourth[mask4] = omega(fourth[mask4])
                denominator4 = (
                    float(energy)
                    - omega_first
                    - omega_second
                    - omega_third
                    - omega_fourth
                )
                integrand4 = (
                    weights[index]
                    * weights[:, None]
                    * weights[None, :]
                    / (
                        16.0
                        * zt**4
                        * omega_first
                        * omega_second
                        * omega_third
                        * omega_fourth
                        * denominator4
                    )
                )
                correction4 += np.sum(integrand4[mask4])
            four[field_index] = (
                frame["quartic"][field_index] ** 2 * correction4 / 24.0
            )
        result = {"three_boson": three, "four_boson": four, "total": three + four}
        self.feshbach = result
        return result

    def rates(
        self,
        potential,
        *,
        inertia=None,
        stiffness=None,
        energy: float = 0.0,
    ):
        r"""Return finite-shell variational-QGRG rates on the field grid."""
        potential = self._field_values(potential, "potential")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self._field_values(inertia, "inertia")
        stiffness = self._field_values(stiffness, "stiffness")
        frame = self.variational_frame(
            potential, inertia=inertia, stiffness=stiffness
        )
        feshbach = self.feshbach_energy(
            frame, inertia=inertia, stiffness=stiffness, energy=energy
        )

        field_dimension = 0.5 * (self.spacetime_dimension - 2.0)
        potential_rate = (
            self.spacetime_dimension * potential
            - field_dimension * self.field * self.derivative(potential, 1)
            + (frame["energy"] + feshbach["total"]) / self.log_width
        )
        denominator = float(energy) - 2.0 * frame["frequency"]
        if np.any(np.isclose(denominator, 0.0)):
            raise ValueError("energy lies on the variational two-boson shell.")
        temporal_integrand = -2.0 * frame["metric_integrand"] / denominator
        temporal_rate = (
            temporal_integrand @ self._momentum_weights / self.log_width
        )
        spring = inertia[:, None] * frame["frequency"] ** 2
        spatial_ratio = stiffness[:, None] * (
            3.0
            - 10.0
            * stiffness[:, None]
            * self._momenta[None, :] ** 2
            / (self.spatial_dimension * spring)
        )
        spatial_rate = (
            (temporal_integrand * spatial_ratio)
            @ self._momentum_weights
            / self.log_width
        )
        inertia_rate = (
            -field_dimension * self.field * self.derivative(inertia, 1)
            + temporal_rate
        )
        stiffness_rate = (
            -field_dimension * self.field * self.derivative(stiffness, 1)
            + spatial_rate
        )
        return potential_rate, inertia_rate, stiffness_rate


class Phi4ContinuousQGRF:
    r"""Continuous 1+1D phi4 QGRF with variational and Feshbach shell response.

    The eliminated Q sector contains states with at least one momentum at the
    running cutoff.  Rates are obtained as logarithmic derivatives of
    cumulative Feshbach integrals over ``max_i |k_i| < cutoff``.
    """

    def __init__(
        self,
        *,
        quadrature_order: int = 40,
        derivative_step: float = 2.0e-3,
        cutoff: float = 1.0,
    ):
        self.quadrature_order = int(quadrature_order)
        self.derivative_step = float(derivative_step)
        self.cutoff = float(cutoff)
        if self.quadrature_order < 8:
            raise ValueError("quadrature_order must be at least 8.")
        if self.derivative_step <= 0.0 or self.cutoff <= 0.0:
            raise ValueError("derivative_step and cutoff must be positive.")
        self._nodes, self._weights = leggauss(self.quadrature_order)
        self.frame = None
        self.components = None

    @staticmethod
    def _variance(mass2: float, cutoff: float) -> float:
        return float(np.arcsinh(cutoff / np.sqrt(mass2)) / (2.0 * pi))

    @classmethod
    def _frame_mass(cls, bare_mass2: float, quartic: float, cutoff: float):
        quartic = float(quartic)
        if quartic < 0.0:
            raise ValueError("quartic must be nonnegative.")
        if quartic == 0.0:
            if bare_mass2 <= 0.0:
                raise ValueError("the free full-cutoff Gaussian frame is unstable.")
            return float(bare_mass2), cls._variance(bare_mass2, cutoff)

        def residual(dressed_mass2):
            return (
                dressed_mass2
                - bare_mass2
                - 0.5 * quartic * cls._variance(dressed_mass2, cutoff)
            )

        lower = 1.0e-13
        upper = max(1.0, bare_mass2 + quartic + cutoff**2)
        for _ in range(80):
            if residual(upper) > 0.0:
                break
            upper *= 2.0
        else:
            raise ValueError("the full-cutoff Hartree gap did not bracket a root.")
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            if residual(midpoint) > 0.0:
                upper = midpoint
            else:
                lower = midpoint
        dressed = 0.5 * (lower + upper)
        return dressed, cls._variance(dressed, cutoff)

    @staticmethod
    def _mass_derivative(field, quartic, dressed_mass2, cutoff):
        derivative_variance = -cutoff / (
            4.0 * pi * dressed_mass2 * np.sqrt(dressed_mass2 + cutoff**2)
        )
        return quartic * field / (1.0 - 0.5 * quartic * derivative_variance)

    def variational_frame(self, field, couplings, *, cutoff=None):
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        field = float(field)
        bare_mass2 = float(Phi4GaussianShell.curvature(field, couplings))
        dressed_mass2, variance = self._frame_mass(
            bare_mass2, couplings.quartic, cutoff
        )
        momentum = cutoff * self._nodes
        weights = cutoff * self._weights / (2.0 * pi)
        frequency = np.sqrt(momentum**2 + dressed_mass2)
        energy = np.sum(
            weights
            * (
                0.25 * frequency
                + (momentum**2 + bare_mass2) / (4.0 * frequency)
            )
        )
        energy += couplings.quartic * variance**2 / 8.0
        return {
            "bare_mass2": bare_mass2,
            "mass2": dressed_mass2,
            "variance": variance,
            "mass_derivative": self._mass_derivative(
                field, couplings.quartic, dressed_mass2, cutoff
            ),
            "energy": float(energy),
        }

    def _triplet_integral(
        self,
        cutoff,
        mass2,
        coupling2,
        *,
        denominator_power: int,
    ):
        momentum1 = cutoff * self._nodes
        weight1 = cutoff * self._weights
        total = 0.0
        for first, first_weight in zip(momentum1, weight1):
            lower = max(-cutoff, -first - cutoff)
            upper = min(cutoff, -first + cutoff)
            if lower >= upper:
                continue
            second = 0.5 * (upper - lower) * self._nodes
            second += 0.5 * (upper + lower)
            second_weight = 0.5 * (upper - lower) * self._weights
            third = -first - second
            omega1 = np.sqrt(first**2 + mass2)
            omega2 = np.sqrt(second**2 + mass2)
            omega3 = np.sqrt(third**2 + mass2)
            gap = omega1 + omega2 + omega3
            total += first_weight * np.sum(
                second_weight
                / (
                    8.0
                    * omega1
                    * omega2
                    * omega3
                    * gap**denominator_power
                )
            )
        return float(coupling2 * total / (6.0 * (2.0 * pi) ** 2))

    def _quartic_energy(self, cutoff, mass2, quartic):
        momentum = cutoff * self._nodes
        weights = cutoff * self._weights
        first = momentum[:, None, None]
        second = momentum[None, :, None]
        third = momentum[None, None, :]
        fourth = -(first + second + third)
        mask = np.abs(fourth) <= cutoff
        omega1 = np.sqrt(first**2 + mass2)
        omega2 = np.sqrt(second**2 + mass2)
        omega3 = np.sqrt(third**2 + mass2)
        omega4 = np.ones_like(fourth)
        omega4[mask] = np.sqrt(fourth[mask] ** 2 + mass2)
        combined_weights = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
        )
        gap = omega1 + omega2 + omega3 + omega4
        integrand = combined_weights / (
            16.0 * omega1 * omega2 * omega3 * omega4 * gap
        )
        return float(
            -quartic**2
            * np.sum(integrand[mask])
            / (24.0 * (2.0 * pi) ** 3)
        )

    def _cumulative_components(
        self,
        field,
        couplings,
        cutoff,
        *,
        fixed_frame=None,
    ):
        if fixed_frame is None:
            frame = self.variational_frame(field, couplings, cutoff=cutoff)
        else:
            frame = dict(fixed_frame)
            frame["variance"] = self._variance(frame["mass2"], cutoff)
            momentum = cutoff * self._nodes
            weights = cutoff * self._weights / (2.0 * pi)
            frequency = np.sqrt(momentum**2 + frame["mass2"])
            frame["energy"] = float(
                np.sum(
                    weights
                    * (
                        0.25 * frequency
                        + (momentum**2 + frame["bare_mass2"])
                        / (4.0 * frequency)
                    )
                )
                + couplings.quartic * frame["variance"] ** 2 / 8.0
            )
        cubic2 = float(Phi4GaussianShell.third_derivative(field, couplings)) ** 2
        energy3 = -self._triplet_integral(
            cutoff,
            frame["mass2"],
            cubic2,
            denominator_power=1,
        )
        energy4 = self._quartic_energy(
            cutoff, frame["mass2"], couplings.quartic
        )
        momentum = cutoff * self._nodes
        weights = cutoff * self._weights / (2.0 * pi)
        pair_temporal = np.sum(
            weights
            * frame["mass_derivative"] ** 2
            / (32.0 * (momentum**2 + frame["mass2"]) ** 2.5)
        )
        triplet_m3 = self._triplet_integral(
            cutoff,
            frame["mass2"],
            couplings.quartic**2,
            denominator_power=3,
        )
        return {
            "frame": frame,
            "gaussian_energy": frame["energy"],
            "three_boson_energy": energy3,
            "four_boson_energy": energy4,
            "total_energy": frame["energy"] + energy3 + energy4,
            "pair_temporal": float(pair_temporal),
            "triplet_temporal": float(2.0 * triplet_m3),
            "total_temporal": float(pair_temporal + 2.0 * triplet_m3),
        }

    def _log_derivative(self, function):
        step = self.derivative_step
        upper = function(self.cutoff * np.exp(step))
        lower = function(self.cutoff * np.exp(-step))
        return (upper - lower) / (2.0 * step)

    def rates(self, field, couplings: Phi4GaussianCouplings):
        r"""Return interacting ``(beta_U, beta_Zt)`` and component diagnostics."""
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("continuous phi4 QGRF requires a Z2-symmetric potential.")
        field = np.asarray(field, dtype=float)
        potential = Phi4GaussianShell.potential(field, couplings)
        potential_rate = np.empty_like(field)
        temporal_rate = np.empty_like(field)
        components = {
            name: np.empty_like(field)
            for name in (
                "hartree_mass2",
                "gaussian_energy",
                "three_boson_energy",
                "four_boson_energy",
                "pair_temporal",
                "triplet_temporal",
                "gaussian_energy_rate",
                "three_boson_energy_rate",
                "four_boson_energy_rate",
                "pair_temporal_rate",
                "triplet_temporal_rate",
            )
        }
        for index in np.ndindex(field.shape):
            value = float(field[index])

            def cumulative(cutoff):
                return self._cumulative_components(value, couplings, cutoff)

            center = cumulative(self.cutoff)
            step = self.derivative_step
            upper = self._cumulative_components(
                value,
                couplings,
                self.cutoff * np.exp(step),
                fixed_frame=center["frame"],
            )
            lower = self._cumulative_components(
                value,
                couplings,
                self.cutoff * np.exp(-step),
                fixed_frame=center["frame"],
            )
            potential_shell = (
                upper["total_energy"] - lower["total_energy"]
            ) / (2.0 * step)
            temporal_shell = (
                upper["total_temporal"] - lower["total_temporal"]
            ) / (2.0 * step)
            potential_rate[index] = 2.0 * potential[index] + potential_shell
            temporal_rate[index] = temporal_shell
            components["hartree_mass2"][index] = center["frame"]["mass2"]
            for name in components:
                if name == "hartree_mass2" or name.endswith("_rate"):
                    continue
                components[name][index] = center[name]
            for name in (
                "gaussian_energy",
                "three_boson_energy",
                "four_boson_energy",
                "pair_temporal",
                "triplet_temporal",
            ):
                components[f"{name}_rate"][index] = (
                    upper[name] - lower[name]
                ) / (2.0 * step)
        self.frame = components["hartree_mass2"]
        self.components = components
        return potential_rate, temporal_rate


class Phi4SmoothQGRF:
    r"""Smooth-filter Hamiltonian QGRF for canonical 1+1D phi4 theory.

    The retained Hamiltonian modes carry the low-pass weight
    ``w_Lambda(k) = exp(-(k / Lambda)^2)``.  A logarithmic cutoff derivative
    of the variational Gaussian energy plus the three- and four-boson
    Feshbach terms gives the potential flow.  Frequency and external spatial
    momentum projections of the same response give independent ``Z_t`` and
    ``Z_x`` flows.  The momentum quadrature is clustered around zero so that
    broken-phase Hartree frames with a small positive gap remain resolved.
    """

    def __init__(
        self,
        *,
        quadrature_order: int = 32,
        momentum_extent: float = 4.5,
        derivative_step: float = 2.0e-3,
        momentum_steps=None,
        cutoff: float = 1.0,
    ):
        self.quadrature_order = int(quadrature_order)
        self.momentum_extent = float(momentum_extent)
        self.derivative_step = float(derivative_step)
        self.cutoff = float(cutoff)
        if self.quadrature_order < 12:
            raise ValueError("quadrature_order must be at least 12.")
        if self.momentum_extent <= 2.0:
            raise ValueError("momentum_extent must exceed 2.")
        if self.derivative_step <= 0.0 or self.cutoff <= 0.0:
            raise ValueError("derivative_step and cutoff must be positive.")
        if momentum_steps is None:
            momentum_steps = (0.006, 0.009, 0.013, 0.018, 0.025)
        self.momentum_steps = np.asarray(momentum_steps, dtype=float)
        if self.momentum_steps.ndim != 1 or self.momentum_steps.size < 3:
            raise ValueError("momentum_steps must contain at least three values.")
        if np.any(self.momentum_steps <= 0.0):
            raise ValueError("momentum_steps must be positive.")
        self._nodes, self._weights = leggauss(self.quadrature_order)
        self.components = None

    @staticmethod
    def _filter(momentum, cutoff):
        return np.exp(-(np.asarray(momentum) / cutoff) ** 2)

    def _grid(self, cutoff):
        bound = self.momentum_extent * cutoff
        coordinate = 0.5 * (self._nodes + 1.0)
        positive = bound * coordinate**2
        positive_weights = bound * coordinate * self._weights
        momentum = np.concatenate((-positive[::-1], positive))
        weights = np.concatenate((positive_weights[::-1], positive_weights))
        return momentum, weights

    def _frame_integrals(self, mass2, cutoff):
        momentum, weights = self._grid(cutoff)
        window = self._filter(momentum, cutoff)
        frequency = np.sqrt(momentum**2 + mass2)
        variance = np.sum(weights * window / (2.0 * frequency)) / (2.0 * pi)
        derivative = -np.sum(weights * window / (4.0 * frequency**3))
        derivative /= 2.0 * pi
        return float(variance), float(derivative)

    def _frame_mass(self, bare_mass2, quartic, cutoff):
        if quartic < 0.0:
            raise ValueError("quartic must be nonnegative.")

        def residual(dressed_mass2):
            variance, _ = self._frame_integrals(dressed_mass2, cutoff)
            return dressed_mass2 - bare_mass2 - 0.5 * quartic * variance

        if quartic == 0.0:
            if bare_mass2 <= 0.0:
                raise ValueError("the smooth free Gaussian frame is unstable.")
            variance, derivative = self._frame_integrals(bare_mass2, cutoff)
            return float(bare_mass2), variance, derivative
        lower = 1.0e-20
        upper = max(1.0, bare_mass2 + quartic + cutoff**2)
        if residual(lower) >= 0.0:
            raise ValueError(
                "the smooth Hartree frame is unresolved near zero mass; "
                "increase quadrature_order"
            )
        for _ in range(80):
            if residual(upper) > 0.0:
                break
            upper *= 2.0
        else:
            raise ValueError("the smooth Hartree gap did not bracket a root.")
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            if residual(midpoint) > 0.0:
                upper = midpoint
            else:
                lower = midpoint
        dressed = 0.5 * (lower + upper)
        variance, derivative = self._frame_integrals(dressed, cutoff)
        return dressed, variance, derivative

    def variational_frame(self, field, couplings, *, cutoff=None):
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        field = float(field)
        bare_mass2 = float(Phi4GaussianShell.curvature(field, couplings))
        mass2, variance, variance_derivative = self._frame_mass(
            bare_mass2, couplings.quartic, cutoff
        )
        mass_derivative = couplings.quartic * field / (
            1.0 - 0.5 * couplings.quartic * variance_derivative
        )
        return {
            "bare_mass2": bare_mass2,
            "mass2": mass2,
            "variance": variance,
            "variance_derivative": variance_derivative,
            "mass_derivative": float(mass_derivative),
        }

    def _gaussian_energy(self, frame, couplings, cutoff):
        momentum, weights = self._grid(cutoff)
        window = self._filter(momentum, cutoff)
        frequency = np.sqrt(momentum**2 + frame["mass2"])
        energy = np.sum(
            weights
            * window
            * (
                0.25 * frequency
                + (momentum**2 + frame["bare_mass2"]) / (4.0 * frequency)
            )
        ) / (2.0 * pi)
        variance = np.sum(weights * window / (2.0 * frequency)) / (2.0 * pi)
        return float(energy + couplings.quartic * variance**2 / 8.0)

    def _triplet_moment(
        self,
        frame,
        coupling2,
        cutoff,
        *,
        denominator_power,
        external_momentum=0.0,
    ):
        momentum, weights = self._grid(cutoff)
        first = momentum[:, None]
        second = momentum[None, :]
        third = -first - second - float(external_momentum)
        omega1 = np.sqrt(first**2 + frame["mass2"])
        omega2 = np.sqrt(second**2 + frame["mass2"])
        omega3 = np.sqrt(third**2 + frame["mass2"])
        window = (
            self._filter(first, cutoff)
            * self._filter(second, cutoff)
            * self._filter(third, cutoff)
        )
        gap = omega1 + omega2 + omega3
        integrand = (
            weights[:, None]
            * weights[None, :]
            * window
            / (
                8.0
                * frame.get("inertia", 1.0) ** 3
                * omega1
                * omega2
                * omega3
                * gap**denominator_power
            )
        )
        return float(coupling2 * np.sum(integrand) / (6.0 * (2.0 * pi) ** 2))

    def _quartic_energy(self, frame, quartic, cutoff):
        momentum, weights = self._grid(cutoff)
        first = momentum[:, None, None]
        second = momentum[None, :, None]
        third = momentum[None, None, :]
        fourth = -(first + second + third)
        omega1 = np.sqrt(first**2 + frame["mass2"])
        omega2 = np.sqrt(second**2 + frame["mass2"])
        omega3 = np.sqrt(third**2 + frame["mass2"])
        omega4 = np.sqrt(fourth**2 + frame["mass2"])
        window = (
            self._filter(first, cutoff)
            * self._filter(second, cutoff)
            * self._filter(third, cutoff)
            * self._filter(fourth, cutoff)
        )
        gap = omega1 + omega2 + omega3 + omega4
        integrand = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * window
            / (16.0 * omega1 * omega2 * omega3 * omega4 * gap)
        )
        return float(
            -quartic**2 * np.sum(integrand) / (24.0 * (2.0 * pi) ** 3)
        )

    def _pair_temporal(self, frame, cutoff):
        momentum, weights = self._grid(cutoff)
        window2 = self._filter(momentum, cutoff) ** 2
        frequency = np.sqrt(momentum**2 + frame["mass2"])
        return float(
            np.sum(
                weights
                * window2
                * frame["mass_derivative"] ** 2
                / (32.0 * frequency**5)
            )
            / (2.0 * pi)
        )

    def _static_response(self, frame, couplings, cutoff, external_momentum):
        momentum, weights = self._grid(cutoff)
        shifted = momentum + float(external_momentum)
        omega = np.sqrt(momentum**2 + frame["mass2"])
        shifted_omega = np.sqrt(shifted**2 + frame["mass2"])
        pair_kernel = (
            self._filter(momentum, cutoff)
            * self._filter(shifted, cutoff)
            / (2.0 * omega * shifted_omega * (omega + shifted_omega))
        )
        pair = -0.5 * frame["mass_derivative"] ** 2
        pair *= np.sum(weights * pair_kernel) / (2.0 * pi)
        triplet = -2.0 * self._triplet_moment(
            frame,
            couplings.quartic**2,
            cutoff,
            denominator_power=1,
            external_momentum=external_momentum,
        )
        return float(pair + triplet)

    def level2_response(self, field, couplings, momentum):
        r"""Return the smooth-weight Gaussian Level-2 QGRG response.

        The low-pass weight is ``exp(-(k / cutoff)^2)``.  Every two-mode
        response carries ``w(k) w(k + q)``, and its shell rate is the centered
        logarithmic cutoff derivative at fixed variational frame.  This is the
        smooth counterpart of :meth:`Phi4GaussianShell.level2_response` and
        contains only the Gaussian pair sector, not the optional three-boson
        Feshbach correction used by :meth:`rates`.
        """
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("smooth Level-2 QGRG requires a Z2-symmetric potential.")
        field = float(field)
        momentum = np.atleast_1d(np.asarray(momentum, dtype=float))
        if momentum.ndim != 1 or not np.all(np.isfinite(momentum)):
            raise ValueError("momentum must be a finite scalar or one-dimensional array.")

        frame = self.variational_frame(field, couplings)

        def cumulative(cutoff, external_momenta):
            internal, weights = self._grid(cutoff)
            frequency = np.sqrt(internal**2 + frame["mass2"])
            window = self._filter(internal, cutoff)
            mass_response = frame["mass_derivative"]
            values = {
                "kernel_response": [],
                "bubble": [],
                "two_point_response": [],
                "overlap_metric": [],
            }
            for external in np.asarray(external_momenta, dtype=float):
                shifted = internal + external
                shifted_frequency = np.sqrt(shifted**2 + frame["mass2"])
                pair_window = window * self._filter(shifted, cutoff)
                measure = weights * pair_window / (2.0 * pi)
                kernel = mass_response * np.sum(
                    measure / (frequency + shifted_frequency)
                )
                bubble = np.sum(
                    measure
                    / (
                        2.0
                        * frequency
                        * shifted_frequency
                        * (frequency + shifted_frequency)
                    )
                )
                metric = mass_response**2 * np.sum(
                    measure
                    / (
                        frequency
                        * shifted_frequency
                        * (frequency + shifted_frequency) ** 2
                    )
                ) / 8.0
                values["kernel_response"].append(kernel)
                values["bubble"].append(bubble)
                values["two_point_response"].append(
                    -0.5 * mass_response**2 * bubble
                )
                values["overlap_metric"].append(metric)
            return {
                name: np.asarray(data, dtype=float) for name, data in values.items()
            }

        scale_step = self.derivative_step
        upper_cutoff = self.cutoff * np.exp(scale_step)
        lower_cutoff = self.cutoff * np.exp(-scale_step)

        def response_at(external_momenta):
            center = cumulative(self.cutoff, external_momenta)
            upper = cumulative(upper_cutoff, external_momenta)
            lower = cumulative(lower_cutoff, external_momenta)
            rates = {
                name: (upper[name] - lower[name]) / (2.0 * scale_step)
                for name in center
            }
            return center, rates

        center, shell = response_at(momentum)
        projection_momenta = np.concatenate(
            ([0.0], self.momentum_steps, -self.momentum_steps)
        )
        _, projection = response_at(projection_momenta)
        size = self.momentum_steps.size
        zero = projection["two_point_response"][0]
        positive = projection["two_point_response"][1 : size + 1]
        negative = projection["two_point_response"][size + 1 :]
        even = 0.5 * (positive + negative)
        slopes = (even - zero) / self.momentum_steps**2
        spatial_rate = np.polynomial.polynomial.polyfit(
            self.momentum_steps**2, slopes, 2
        )[0]

        upper_temporal = self._pair_temporal(frame, upper_cutoff)
        lower_temporal = self._pair_temporal(frame, lower_cutoff)
        temporal_rate = (upper_temporal - lower_temporal) / (2.0 * scale_step)
        shell["two_point_rate"] = shell.pop("two_point_response")
        return {
            "momentum": momentum.copy(),
            "frequency": float(np.sqrt(self.cutoff**2 + frame["mass2"])),
            "frame_mass2": float(frame["mass2"]),
            "mass_response": float(frame["mass_derivative"]),
            "kernel_response": shell["kernel_response"],
            "bubble": shell["bubble"],
            "two_point_rate": shell["two_point_rate"],
            "overlap_metric": shell["overlap_metric"],
            "temporal_rate": float(temporal_rate),
            "spatial_rate": float(spatial_rate),
            "cumulative_kernel_response": center["kernel_response"],
            "cumulative_bubble": center["bubble"],
            "cumulative_two_point_response": center["two_point_response"],
            "cumulative_overlap_metric": center["overlap_metric"],
        }

    def _cumulative_components(self, field, couplings, cutoff, *, frame):
        gaussian_energy = self._gaussian_energy(frame, couplings, cutoff)
        cubic2 = float(Phi4GaussianShell.third_derivative(field, couplings)) ** 2
        energy3 = -self._triplet_moment(
            frame, cubic2, cutoff, denominator_power=1
        )
        energy4 = self._quartic_energy(frame, couplings.quartic, cutoff)
        pair_temporal = self._pair_temporal(frame, cutoff)
        triplet_temporal = 2.0 * self._triplet_moment(
            frame,
            couplings.quartic**2,
            cutoff,
            denominator_power=3,
        )
        return {
            "gaussian_energy": gaussian_energy,
            "three_boson_energy": energy3,
            "four_boson_energy": energy4,
            "total_energy": gaussian_energy + energy3 + energy4,
            "pair_temporal": pair_temporal,
            "triplet_temporal": triplet_temporal,
            "total_temporal": pair_temporal + triplet_temporal,
        }

    def rates(self, field, couplings: Phi4GaussianCouplings):
        r"""Return smooth-filter ``(beta_U, beta_Zt, beta_Zx)``."""
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("smooth phi4 QGRF requires a Z2-symmetric potential.")
        field = np.asarray(field, dtype=float)
        potential = Phi4GaussianShell.potential(field, couplings)
        potential_rate = np.empty_like(field)
        temporal_rate = np.empty_like(field)
        spatial_rate = np.empty_like(field)
        components = {
            name: np.empty_like(field)
            for name in (
                "hartree_mass2",
                "pair_temporal_rate",
                "triplet_temporal_rate",
                "three_boson_energy_rate",
                "four_boson_energy_rate",
            )
        }
        scale_step = self.derivative_step
        upper_cutoff = self.cutoff * np.exp(scale_step)
        lower_cutoff = self.cutoff * np.exp(-scale_step)
        for index in np.ndindex(field.shape):
            value = float(field[index])
            frame = self.variational_frame(value, couplings)
            upper = self._cumulative_components(
                value, couplings, upper_cutoff, frame=frame
            )
            lower = self._cumulative_components(
                value, couplings, lower_cutoff, frame=frame
            )
            potential_shell = (
                upper["total_energy"] - lower["total_energy"]
            ) / (2.0 * scale_step)
            potential_rate[index] = 2.0 * potential[index] + potential_shell
            temporal_rate[index] = (
                upper["total_temporal"] - lower["total_temporal"]
            ) / (2.0 * scale_step)

            static_shell = []
            for momentum_step in np.concatenate(([0.0], self.momentum_steps)):
                if momentum_step == 0.0:
                    upper_static = self._static_response(
                        frame, couplings, upper_cutoff, 0.0
                    )
                    lower_static = self._static_response(
                        frame, couplings, lower_cutoff, 0.0
                    )
                else:
                    upper_static = 0.5 * (
                        self._static_response(
                            frame, couplings, upper_cutoff, momentum_step
                        )
                        + self._static_response(
                            frame, couplings, upper_cutoff, -momentum_step
                        )
                    )
                    lower_static = 0.5 * (
                        self._static_response(
                            frame, couplings, lower_cutoff, momentum_step
                        )
                        + self._static_response(
                            frame, couplings, lower_cutoff, -momentum_step
                        )
                    )
                static_shell.append(
                    (upper_static - lower_static) / (2.0 * scale_step)
                )
            static_shell = np.asarray(static_shell)
            slopes = (static_shell[1:] - static_shell[0]) / self.momentum_steps**2
            spatial_rate[index] = np.polynomial.polynomial.polyfit(
                self.momentum_steps**2, slopes, 2
            )[0]
            components["hartree_mass2"][index] = frame["mass2"]
            for name in (
                "pair_temporal",
                "triplet_temporal",
                "three_boson_energy",
                "four_boson_energy",
            ):
                components[f"{name}_rate"][index] = (
                    upper[name] - lower[name]
                ) / (2.0 * scale_step)
        self.components = components
        return potential_rate, temporal_rate, spatial_rate


@dataclass(frozen=True)
class ExponentialRegulator:
    r"""Smooth covariant regulator ``R(y) = y / (exp(y) - 1)``."""

    @staticmethod
    def value(momentum2):
        momentum2 = np.asarray(momentum2, dtype=float)
        out = np.empty_like(momentum2)
        small = np.abs(momentum2) < 1.0e-5
        large = momentum2 > 50.0
        regular = ~(small | large)
        value = momentum2[small]
        out[small] = (
            1.0 - value / 2.0 + value**2 / 12.0 - value**4 / 720.0
        )
        value = momentum2[regular]
        out[regular] = value / np.expm1(value)
        out[large] = 0.0
        return out

    @staticmethod
    def first_derivative(momentum2):
        r"""Return ``d R(y) / d y``."""
        momentum2 = np.asarray(momentum2, dtype=float)
        out = np.empty_like(momentum2)
        small = np.abs(momentum2) < 1.0e-4
        large = momentum2 > 50.0
        regular = ~(small | large)
        value = momentum2[small]
        out[small] = -0.5 + value / 6.0 - value**3 / 180.0
        value = momentum2[regular]
        exponential = np.exp(value)
        denominator = np.expm1(value)
        out[regular] = (
            exponential * (1.0 - value) - 1.0
        ) / denominator**2
        out[large] = 0.0
        return out

    @staticmethod
    def second_derivative(momentum2):
        r"""Return ``d^2 R(y) / d y^2``."""
        momentum2 = np.asarray(momentum2, dtype=float)
        out = np.empty_like(momentum2)
        small = np.abs(momentum2) < 1.0e-4
        large = momentum2 > 50.0
        regular = ~(small | large)
        value = momentum2[small]
        out[small] = 1.0 / 6.0 - value**2 / 60.0 + value**4 / 1008.0
        value = momentum2[regular]
        exponential = np.exp(value)
        denominator = np.expm1(value)
        numerator = exponential * (1.0 - value) - 1.0
        out[regular] = (
            -value * exponential * denominator
            - 2.0 * numerator * exponential
        ) / denominator**3
        out[large] = 0.0
        return out

    def scale_derivative(self, momentum2, anomalous_dimension=0.0):
        r"""Return ``partial_t R`` at fixed dimensionful momentum."""
        momentum2 = np.asarray(momentum2, dtype=float)
        eta = float(anomalous_dimension)
        out = np.empty_like(momentum2)
        small = np.abs(momentum2) < 1.0e-5
        large = momentum2 > 50.0
        regular = ~(small | large)
        value = momentum2[small]
        regulator = (
            1.0 - value / 2.0 + value**2 / 12.0 - value**4 / 720.0
        )
        base = 2.0 - value**2 / 6.0 + value**4 / 120.0
        out[small] = base - eta * regulator
        value = momentum2[regular]
        denominator = np.expm1(value)
        base = 2.0 * value**2 * np.exp(value) / denominator**2
        out[regular] = base - eta * value / denominator
        out[large] = 0.0
        return out


@dataclass(frozen=True)
class GaussianRegulator:
    r"""Smooth regulator ``R_k(p) = Z k^2 exp(-p^2/k^2)``."""

    @staticmethod
    def value(momentum2):
        momentum2 = np.asarray(momentum2, dtype=float)
        return np.exp(-momentum2)

    @staticmethod
    def first_derivative(momentum2):
        r"""Return ``d R(y) / d y``."""
        momentum2 = np.asarray(momentum2, dtype=float)
        return -np.exp(-momentum2)

    @staticmethod
    def second_derivative(momentum2):
        r"""Return ``d^2 R(y) / d y^2``."""
        momentum2 = np.asarray(momentum2, dtype=float)
        return np.exp(-momentum2)

    @staticmethod
    def scale_derivative(momentum2, anomalous_dimension=0.0):
        r"""Return ``partial_t R`` at fixed dimensionful momentum."""
        momentum2 = np.asarray(momentum2, dtype=float)
        return (
            2.0 + 2.0 * momentum2 - float(anomalous_dimension)
        ) * np.exp(-momentum2)


class Phi4RegulatedQGRF:
    r"""Regulator-based Hamiltonian QGRF for canonical 1+1D phi4.

    A smooth infrared regulator enters the fluctuation frequencies rather
    than being treated as a Hilbert-space projector.  The Gaussian potential
    flow is the exact regulated one-loop threshold.  The optional Feshbach
    closure adds regulator derivatives of the normal-ordered three- and
    four-boson residual energies.  Temporal and spatial kinetic flows are
    projected from the corresponding regulated resolvent kernels.
    """

    def __init__(
        self,
        *,
        quadrature_order: int = 32,
        momentum_scale: float = 1.0,
        projection_extent: float = 8.0,
        momentum_steps=None,
        cutoff: float = 1.0,
        regulator=None,
        include_feshbach: bool = True,
        feshbach_strength: float = 1.0,
    ):
        self.quadrature_order = int(quadrature_order)
        self.momentum_scale = float(momentum_scale)
        self.projection_extent = float(projection_extent)
        self.cutoff = float(cutoff)
        self.regulator = ExponentialRegulator() if regulator is None else regulator
        self.include_feshbach = bool(include_feshbach)
        self.feshbach_strength = (
            float(feshbach_strength) if self.include_feshbach else 0.0
        )
        if self.quadrature_order < 12:
            raise ValueError("quadrature_order must be at least 12.")
        if self.momentum_scale <= 0.0:
            raise ValueError("momentum_scale must be positive.")
        if self.projection_extent <= 3.0:
            raise ValueError("projection_extent must exceed 3.")
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive.")
        if not 0.0 <= self.feshbach_strength <= 1.0:
            raise ValueError("feshbach_strength must lie between zero and one.")
        if momentum_steps is None:
            momentum_steps = (0.006, 0.009, 0.013, 0.018, 0.025)
        self.momentum_steps = np.asarray(momentum_steps, dtype=float)
        if self.momentum_steps.ndim != 1 or self.momentum_steps.size < 3:
            raise ValueError("momentum_steps must contain at least three values.")
        if np.any(self.momentum_steps <= 0.0):
            raise ValueError("momentum_steps must be positive.")
        nodes, weights = leggauss(self.quadrature_order)
        coordinate = 0.5 * (nodes + 1.0)
        denominator = 1.0 - coordinate**2
        scale = self.momentum_scale * self.cutoff
        positive = scale * coordinate**2 / denominator
        positive_weights = scale * coordinate * weights / denominator**2
        self._momentum = np.concatenate((-positive[::-1], positive))
        self._weights = np.concatenate(
            (positive_weights[::-1], positive_weights)
        )
        projection = self.projection_extent * self.cutoff * coordinate**2
        projection_weights = (
            self.projection_extent * self.cutoff * coordinate * weights
        )
        self._projection_momentum = np.concatenate(
            (-projection[::-1], projection)
        )
        self._projection_weights = np.concatenate(
            (projection_weights[::-1], projection_weights)
        )
        self.components = None

    def set_feshbach_strength(self, strength):
        """Set the homotopy strength of the residual Feshbach functional."""
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("feshbach strength must lie between zero and one.")
        self.feshbach_strength = strength if self.include_feshbach else 0.0
        return self

    def _regulator_value(self, momentum, cutoff):
        cutoff = float(cutoff)
        momentum2 = np.asarray(momentum, dtype=float) ** 2
        return cutoff**2 * self.regulator.value(momentum2 / cutoff**2)

    def _frequency(
        self, momentum, curvature, cutoff, *, inertia=1.0, stiffness=1.0
    ):
        momentum = np.asarray(momentum, dtype=float)
        inertia = float(inertia)
        stiffness = float(stiffness)
        if inertia <= 0.0 or stiffness <= 0.0:
            raise ValueError("inertia and stiffness must be positive.")
        frequency2 = (
            stiffness * momentum**2
            + self._regulator_value(momentum, cutoff)
            + float(curvature)
        ) / inertia
        if np.any(frequency2 <= 0.0):
            raise ValueError("the regulated Hamiltonian frame is unstable.")
        return np.sqrt(frequency2)

    def _frequency_rate(
        self, momentum, curvature, cutoff, *, inertia=1.0, stiffness=1.0
    ):
        """Return the regulated frequency and its single-scale IR rate."""
        momentum = np.asarray(momentum, dtype=float)
        frequency = self._frequency(
            momentum,
            curvature,
            cutoff,
            inertia=inertia,
            stiffness=stiffness,
        )
        momentum2 = momentum**2
        regulator_rate = -cutoff**2 * self.regulator.scale_derivative(
            momentum2 / cutoff**2
        )
        return frequency, regulator_rate / (2.0 * inertia * frequency)

    def frame(self, field, couplings, *, cutoff=None):
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        curvature = float(Phi4GaussianShell.curvature(field, couplings))
        gap2 = cutoff**2 + curvature
        if gap2 <= 0.0:
            raise ValueError("the regulated Hamiltonian frame is unstable.")
        return {
            "curvature": curvature,
            "gap2": gap2,
            "inertia": 1.0,
            "stiffness": 1.0,
            "curvature_derivative": float(
                Phi4GaussianShell.third_derivative(field, couplings)
            ),
        }

    def _frame_frequency(self, momentum, frame, cutoff):
        return self._frequency(
            momentum,
            frame["curvature"],
            cutoff,
            inertia=frame.get("inertia", 1.0),
            stiffness=frame.get("stiffness", 1.0),
        )

    def _frame_frequency_rate(self, momentum, frame, cutoff):
        return self._frequency_rate(
            momentum,
            frame["curvature"],
            cutoff,
            inertia=frame.get("inertia", 1.0),
            stiffness=frame.get("stiffness", 1.0),
        )

    def _gaussian_loop(self, frame):
        momentum2 = self._momentum**2
        cutoff = self.cutoff
        scale_derivative = cutoff**2 * self.regulator.scale_derivative(
            momentum2 / cutoff**2
        )
        frequency = self._frequency(
            self._momentum,
            frame["curvature"],
            cutoff,
            inertia=frame.get("inertia", 1.0),
            stiffness=frame.get("stiffness", 1.0),
        )
        return float(
            -np.sum(
                self._weights
                * scale_derivative
                / (4.0 * frame.get("inertia", 1.0) * frequency)
            )
            / (2.0 * pi)
        )

    def _triplet_moment(
        self,
        frame,
        coupling2,
        cutoff,
        *,
        denominator_power,
        external_momentum=0.0,
    ):
        first = self._momentum[:, None]
        second = self._momentum[None, :]
        third = -first - second - float(external_momentum)
        omega1 = self._frame_frequency(first, frame, cutoff)
        omega2 = self._frame_frequency(second, frame, cutoff)
        omega3 = self._frame_frequency(third, frame, cutoff)
        gap = omega1 + omega2 + omega3
        integrand = (
            self._weights[:, None]
            * self._weights[None, :]
            / (
                8.0
                * frame.get("inertia", 1.0) ** 3
                * omega1
                * omega2
                * omega3
                * gap**denominator_power
            )
        )
        return float(coupling2 * np.sum(integrand) / (6.0 * (2.0 * pi) ** 2))

    def _triplet_moment_rate(
        self,
        frame,
        coupling2,
        cutoff,
        *,
        denominator_power,
        external_momentum=0.0,
    ):
        first = self._momentum[:, None]
        second = self._momentum[None, :]
        third = -first - second - float(external_momentum)
        omega1, omega1_rate = self._frame_frequency_rate(first, frame, cutoff)
        omega2, omega2_rate = self._frame_frequency_rate(second, frame, cutoff)
        omega3, omega3_rate = self._frame_frequency_rate(third, frame, cutoff)
        gap = omega1 + omega2 + omega3
        gap_rate = omega1_rate + omega2_rate + omega3_rate
        integrand = (
            self._weights[:, None]
            * self._weights[None, :]
            / (
                8.0
                * frame.get("inertia", 1.0) ** 3
                * omega1
                * omega2
                * omega3
                * gap**denominator_power
            )
        )
        logarithmic_rate = (
            omega1_rate / omega1
            + omega2_rate / omega2
            + omega3_rate / omega3
            + denominator_power * gap_rate / gap
        )
        return float(
            -coupling2
            * np.sum(integrand * logarithmic_rate)
            / (6.0 * (2.0 * pi) ** 2)
        )

    def _quartic_energy(self, frame, quartic, cutoff):
        first = self._momentum[:, None, None]
        second = self._momentum[None, :, None]
        third = self._momentum[None, None, :]
        fourth = -(first + second + third)
        omega1 = self._frame_frequency(first, frame, cutoff)
        omega2 = self._frame_frequency(second, frame, cutoff)
        omega3 = self._frame_frequency(third, frame, cutoff)
        omega4 = self._frame_frequency(fourth, frame, cutoff)
        gap = omega1 + omega2 + omega3 + omega4
        integrand = (
            self._weights[:, None, None]
            * self._weights[None, :, None]
            * self._weights[None, None, :]
            / (
                16.0
                * frame.get("inertia", 1.0) ** 4
                * omega1
                * omega2
                * omega3
                * omega4
                * gap
            )
        )
        return float(
            -quartic**2 * np.sum(integrand) / (24.0 * (2.0 * pi) ** 3)
        )

    def _triplet_energy_rate(self, frame, coupling2, cutoff):
        return -self._triplet_moment_rate(
            frame,
            coupling2,
            cutoff,
            denominator_power=1,
        )

    def _quartic_energy_rate(self, frame, quartic, cutoff):
        first = self._momentum[:, None, None]
        second = self._momentum[None, :, None]
        third = self._momentum[None, None, :]
        fourth = -(first + second + third)
        omega1, omega1_rate = self._frame_frequency_rate(first, frame, cutoff)
        omega2, omega2_rate = self._frame_frequency_rate(second, frame, cutoff)
        omega3, omega3_rate = self._frame_frequency_rate(third, frame, cutoff)
        omega4, omega4_rate = self._frame_frequency_rate(fourth, frame, cutoff)
        gap = omega1 + omega2 + omega3 + omega4
        gap_rate = omega1_rate + omega2_rate + omega3_rate + omega4_rate
        integrand = (
            self._weights[:, None, None]
            * self._weights[None, :, None]
            * self._weights[None, None, :]
            / (
                16.0
                * frame.get("inertia", 1.0) ** 4
                * omega1
                * omega2
                * omega3
                * omega4
                * gap
            )
        )
        logarithmic_rate = (
            omega1_rate / omega1
            + omega2_rate / omega2
            + omega3_rate / omega3
            + omega4_rate / omega4
            + gap_rate / gap
        )
        return float(
            quartic**2
            * np.sum(integrand * logarithmic_rate)
            / (24.0 * (2.0 * pi) ** 3)
        )

    def _pair_temporal(self, frame, cutoff):
        frequency = self._frame_frequency(self._momentum, frame, cutoff)
        return float(
            np.sum(
                self._weights
                * frame["curvature_derivative"] ** 2
                / (32.0 * frame.get("inertia", 1.0) ** 2 * frequency**5)
            )
            / (2.0 * pi)
        )

    def _pair_temporal_rate(self, frame, cutoff):
        frequency, frequency_rate = self._frame_frequency_rate(
            self._momentum, frame, cutoff
        )
        integrand = (
            self._weights
            * frame["curvature_derivative"] ** 2
            / (32.0 * frame.get("inertia", 1.0) ** 2 * frequency**5)
        )
        return float(
            -5.0
            * np.sum(integrand * frequency_rate / frequency)
            / (2.0 * pi)
        )

    def _static_response(
        self, frame, couplings, cutoff, external_momentum
    ):
        momentum = self._projection_momentum
        weights = self._projection_weights
        shifted = momentum + float(external_momentum)
        omega = self._frame_frequency(momentum, frame, cutoff)
        shifted_omega = self._frame_frequency(shifted, frame, cutoff)
        pair_kernel = 1.0 / (
            2.0
            * frame.get("inertia", 1.0) ** 2
            * omega
            * shifted_omega
            * (omega + shifted_omega)
        )
        pair = -0.5 * frame["curvature_derivative"] ** 2
        pair *= np.sum(weights * pair_kernel) / (2.0 * pi)
        triplet = 0.0
        if self.feshbach_strength:
            first = momentum[:, None]
            second = momentum[None, :]
            third = -first - second - float(external_momentum)
            omega1 = self._frame_frequency(first, frame, cutoff)
            omega2 = self._frame_frequency(second, frame, cutoff)
            omega3 = self._frame_frequency(third, frame, cutoff)
            gap = omega1 + omega2 + omega3
            moment = couplings.quartic**2 * np.sum(
                weights[:, None]
                * weights[None, :]
                / (
                    8.0
                    * frame.get("inertia", 1.0) ** 3
                    * omega1
                    * omega2
                    * omega3
                    * gap
                )
            )
            moment /= 6.0 * (2.0 * pi) ** 2
            triplet = -2.0 * self.feshbach_strength * moment
        return float(pair + triplet)

    def _static_response_rate(
        self, frame, couplings, cutoff, external_momentum
    ):
        momentum = self._projection_momentum
        weights = self._projection_weights
        shifted = momentum + float(external_momentum)
        omega, omega_rate = self._frame_frequency_rate(momentum, frame, cutoff)
        shifted_omega, shifted_rate = self._frame_frequency_rate(
            shifted, frame, cutoff
        )
        gap = omega + shifted_omega
        pair_kernel = 1.0 / (
            2.0
            * frame.get("inertia", 1.0) ** 2
            * omega
            * shifted_omega
            * gap
        )
        pair_logarithmic_rate = (
            omega_rate / omega
            + shifted_rate / shifted_omega
            + (omega_rate + shifted_rate) / gap
        )
        pair_rate = 0.5 * frame["curvature_derivative"] ** 2
        pair_rate *= (
            np.sum(weights * pair_kernel * pair_logarithmic_rate)
            / (2.0 * pi)
        )

        triplet_rate = 0.0
        if self.feshbach_strength:
            first = momentum[:, None]
            second = momentum[None, :]
            third = -first - second - float(external_momentum)
            omega1, omega1_rate = self._frame_frequency_rate(
                first, frame, cutoff
            )
            omega2, omega2_rate = self._frame_frequency_rate(
                second, frame, cutoff
            )
            omega3, omega3_rate = self._frame_frequency_rate(
                third, frame, cutoff
            )
            gap = omega1 + omega2 + omega3
            gap_rate = omega1_rate + omega2_rate + omega3_rate
            integrand = (
                weights[:, None]
                * weights[None, :]
                / (
                    8.0
                    * frame.get("inertia", 1.0) ** 3
                    * omega1
                    * omega2
                    * omega3
                    * gap
                )
            )
            logarithmic_rate = (
                omega1_rate / omega1
                + omega2_rate / omega2
                + omega3_rate / omega3
                + gap_rate / gap
            )
            triplet_rate = self.feshbach_strength * couplings.quartic**2 * np.sum(
                integrand * logarithmic_rate
            )
            triplet_rate /= 3.0 * (2.0 * pi) ** 2
        return float(pair_rate + triplet_rate)

    def potential_rate(self, field, couplings: Phi4GaussianCouplings):
        """Return the regulated potential flow without kinetic projections."""
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("regulated phi4 QGRF requires a Z2-symmetric potential.")
        if couplings.quartic < 0.0:
            raise ValueError("quartic must be nonnegative.")
        field = np.asarray(field, dtype=float)
        potential = Phi4GaussianShell.potential(field, couplings)
        out = np.empty_like(field)
        for index in np.ndindex(field.shape):
            value = float(field[index])
            frame = self.frame(value, couplings)
            residual_rate = 0.0
            if self.feshbach_strength:
                cubic2 = float(
                    Phi4GaussianShell.third_derivative(value, couplings)
                ) ** 2
                residual_rate += self.feshbach_strength * self._triplet_energy_rate(
                    frame, cubic2, self.cutoff
                )
                residual_rate += self.feshbach_strength * self._quartic_energy_rate(
                    frame, couplings.quartic, self.cutoff
                )
            out[index] = (
                2.0 * potential[index]
                + self._gaussian_loop(frame)
                + residual_rate
            )
        return out

    def rates(self, field, couplings: Phi4GaussianCouplings):
        r"""Return raw ``(beta_U, beta_Zt, beta_Zx)`` before rescaling."""
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("regulated phi4 QGRF requires a Z2-symmetric potential.")
        if couplings.quartic < 0.0:
            raise ValueError("quartic must be nonnegative.")
        field = np.asarray(field, dtype=float)
        potential = Phi4GaussianShell.potential(field, couplings)
        potential_rate = np.empty_like(field)
        temporal_rate = np.empty_like(field)
        spatial_rate = np.empty_like(field)
        components = {
            name: np.empty_like(field)
            for name in (
                "regulated_gap2",
                "gaussian_loop_rate",
                "three_boson_energy_rate",
                "four_boson_energy_rate",
                "pair_temporal_rate",
                "triplet_temporal_rate",
            )
        }
        for index in np.ndindex(field.shape):
            value = float(field[index])
            frame = self.frame(value, couplings)
            if self.feshbach_strength:
                cubic2 = float(
                    Phi4GaussianShell.third_derivative(value, couplings)
                ) ** 2
                three_boson_rate = self.feshbach_strength * self._triplet_energy_rate(
                    frame, cubic2, self.cutoff
                )
                four_boson_rate = self.feshbach_strength * self._quartic_energy_rate(
                    frame, couplings.quartic, self.cutoff
                )
            else:
                three_boson_rate = 0.0
                four_boson_rate = 0.0
            residual_energy_rate = three_boson_rate + four_boson_rate
            gaussian_loop = self._gaussian_loop(frame)
            potential_rate[index] = (
                2.0 * potential[index]
                + gaussian_loop
                + residual_energy_rate
            )
            pair_temporal_rate = self._pair_temporal_rate(
                frame, self.cutoff
            )
            if self.feshbach_strength:
                triplet_temporal_rate = 2.0 * self.feshbach_strength * self._triplet_moment_rate(
                    frame,
                    couplings.quartic**2,
                    self.cutoff,
                    denominator_power=3,
                )
            else:
                triplet_temporal_rate = 0.0
            temporal_rate[index] = (
                pair_temporal_rate + triplet_temporal_rate
            )

            static_rates = []
            for momentum_step in np.concatenate(([0.0], self.momentum_steps)):
                if momentum_step == 0.0:
                    static_rate = self._static_response_rate(
                        frame, couplings, self.cutoff, 0.0
                    )
                else:
                    static_rate = 0.5 * (
                        self._static_response_rate(
                            frame, couplings, self.cutoff, momentum_step
                        )
                        + self._static_response_rate(
                            frame, couplings, self.cutoff, -momentum_step
                        )
                    )
                static_rates.append(static_rate)
            static_rates = np.asarray(static_rates)
            slopes = (
                static_rates[1:] - static_rates[0]
            ) / self.momentum_steps**2
            spatial_rate[index] = np.polynomial.polynomial.polyfit(
                self.momentum_steps**2, slopes, 2
            )[0]
            components["regulated_gap2"][index] = frame["gap2"]
            components["gaussian_loop_rate"][index] = gaussian_loop
            components["three_boson_energy_rate"][index] = three_boson_rate
            components["four_boson_energy_rate"][index] = four_boson_rate
            components["pair_temporal_rate"][index] = pair_temporal_rate
            components["triplet_temporal_rate"][index] = (
                triplet_temporal_rate
            )
        self.components = components
        return potential_rate, temporal_rate, spatial_rate

    def beta(self, couplings: Phi4GaussianCouplings, *, radius=0.08):
        r"""Return normalized quartic flow, ``eta_t``, ``eta_x``, and ``z``."""
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        field = np.linspace(-radius, radius, 11)
        potential_rate = self.potential_rate(field, couplings)
        coefficients = np.polynomial.polynomial.polyfit(
            field / radius, potential_rate, 8
        )
        mass2 = 2.0 * coefficients[2] / radius**2
        quartic = 24.0 * coefficients[4] / radius**4
        minimum = (
            np.sqrt(-6.0 * couplings.mass2 / couplings.quartic)
            if couplings.mass2 < 0.0 and couplings.quartic > 0.0
            else 0.0
        )
        _, temporal, spatial = self.rates(np.array([minimum]), couplings)
        eta_t = float(temporal[0])
        eta_x = float(spatial[0])
        dynamic_exponent = 1.0 + 0.5 * (eta_t - eta_x)
        mass2 -= eta_x * couplings.mass2
        quartic -= (
            dynamic_exponent - 1.0 + 2.0 * eta_x
        ) * couplings.quartic
        return (
            Phi4GaussianCouplings(mass2=mass2, quartic=quartic),
            eta_t,
            eta_x,
            dynamic_exponent,
        )


class Phi4FunctionalRegulatedQGRF:
    r"""Functional regulator-based QGRF on a field grid.

    ``U(phi)`` and its first four derivatives are represented directly on the
    grid. At each field point the regulated shell Hamiltonian uses the local
    ``U''``, ``U'''``, ``U''''``, ``Z_t``, and ``Z_x``. The Gaussian shell
    determinant and residual Feshbach vacuum channels enter once each, while
    external-frequency and external-momentum projections generate
    field-resolved kinetic sources.
    """

    def __init__(
        self,
        field,
        *,
        stencil_size: int = 9,
        normalize_index: int | None = None,
        kinetic_strength: float = 1.0,
        **kernel_options,
    ):
        self.field = np.asarray(field, dtype=float)
        self.grid = Phi4FunctionalQGRG(
            self.field, spatial_dimension=1, stencil_size=stencil_size
        )
        self.kernel = Phi4RegulatedQGRF(**kernel_options)
        if normalize_index is None:
            normalize_index = int(np.argmin(np.abs(self.field)))
        self.normalize_index = int(normalize_index)
        if not 0 <= self.normalize_index < self.field.size:
            raise ValueError("normalize_index lies outside the field grid.")
        self.kinetic_strength = float(kinetic_strength)
        if not 0.0 <= self.kinetic_strength <= 1.0:
            raise ValueError("kinetic_strength must lie between zero and one.")
        self.geometry = None
        self.fixed_potential = None
        self.fixed_inertia = None
        self.fixed_stiffness = None
        self.fixed_beta = None
        self.self_energy = None

    def set_kinetic_strength(self, strength):
        """Set the homotopy strength of the geometric kinetic feedback."""
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("kinetic strength must lie between zero and one.")
        self.kinetic_strength = strength
        return self

    def derivative(self, values, order: int):
        return self.grid.derivative(values, order)

    def _point_sources(
        self,
        curvature,
        cubic,
        quartic,
        *,
        kinetic,
        inertia=1.0,
        stiffness=1.0,
    ):
        frame = {
            "curvature": float(curvature),
            "gap2": self.kernel.cutoff**2 + float(curvature),
            "curvature_derivative": float(cubic),
            "inertia": float(inertia),
            "stiffness": float(stiffness),
        }
        if frame["gap2"] <= 0.0 or inertia <= 0.0 or stiffness <= 0.0:
            raise ValueError("the regulated functional frame is unstable.")
        potential = self.kernel._gaussian_loop(frame)
        if self.kernel.feshbach_strength:
            potential += self.kernel.feshbach_strength * self.kernel._triplet_energy_rate(
                frame, float(cubic) ** 2, self.kernel.cutoff
            )
            potential += self.kernel.feshbach_strength * self.kernel._quartic_energy_rate(
                frame, float(quartic), self.kernel.cutoff
            )
        if not kinetic:
            return potential, 0.0, 0.0

        temporal = self.kernel._pair_temporal_rate(
            frame, self.kernel.cutoff
        )
        if self.kernel.feshbach_strength:
            temporal += 2.0 * self.kernel.feshbach_strength * self.kernel._triplet_moment_rate(
                frame,
                float(quartic) ** 2,
                self.kernel.cutoff,
                denominator_power=3,
            )
        local_couplings = Phi4GaussianCouplings(quartic=float(quartic))
        static_rates = []
        for momentum_step in np.concatenate(
            ([0.0], self.kernel.momentum_steps)
        ):
            if momentum_step == 0.0:
                static_rate = self.kernel._static_response_rate(
                    frame, local_couplings, self.kernel.cutoff, 0.0
                )
            else:
                static_rate = 0.5 * (
                    self.kernel._static_response_rate(
                        frame,
                        local_couplings,
                        self.kernel.cutoff,
                        momentum_step,
                    )
                    + self.kernel._static_response_rate(
                        frame,
                        local_couplings,
                        self.kernel.cutoff,
                        -momentum_step,
                    )
                )
            static_rates.append(static_rate)
        static_rates = np.asarray(static_rates)
        slopes = (
            static_rates[1:] - static_rates[0]
        ) / self.kernel.momentum_steps**2
        spatial = np.polynomial.polynomial.polyfit(
            self.kernel.momentum_steps**2, slopes, 2
        )[0]
        return potential, temporal, float(spatial)

    def sources(self, potential, *, inertia=None, stiffness=None, kinetic=True):
        """Return local loop sources before canonical field rescaling."""
        potential = self.grid._field_values(potential, "potential")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self.grid._field_values(inertia, "inertia")
        stiffness = self.grid._field_values(stiffness, "stiffness")
        curvature = self.derivative(potential, 2)
        cubic = self.derivative(potential, 3)
        quartic = self.derivative(potential, 4)
        potential_source = np.empty_like(self.field)
        temporal_source = np.empty_like(self.field)
        spatial_source = np.empty_like(self.field)
        for index in range(self.field.size):
            values = self._point_sources(
                curvature[index],
                cubic[index],
                quartic[index],
                kinetic=kinetic,
                inertia=inertia[index],
                stiffness=stiffness[index],
            )
            potential_source[index] = values[0]
            temporal_source[index] = values[1]
            spatial_source[index] = values[2]
        self.geometry = {
            "curvature": curvature,
            "cubic": cubic,
            "quartic": quartic,
            "potential_source": potential_source,
            "temporal_source": temporal_source,
            "spatial_source": spatial_source,
        }
        return potential_source, temporal_source, spatial_source

    def rates(self, potential, *, inertia=None, stiffness=None):
        r"""Return functional ``(beta_U, beta_Zt, beta_Zx)`` on the grid."""
        potential = self.grid._field_values(potential, "potential")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self.grid._field_values(inertia, "inertia")
        stiffness = self.grid._field_values(stiffness, "stiffness")
        potential_source, temporal_source, spatial_source = self.sources(
            potential,
            inertia=inertia,
            stiffness=stiffness,
            kinetic=True,
        )
        temporal_source = self.kinetic_strength * temporal_source
        spatial_source = self.kinetic_strength * spatial_source
        center = self.normalize_index
        eta_t = temporal_source[center] / inertia[center]
        eta_x = spatial_source[center] / stiffness[center]
        dynamic_exponent = 1.0 + 0.5 * (eta_t - eta_x)
        field_dimension = 0.5 * (
            dynamic_exponent - 1.0 + eta_x
        )
        potential_rate = (
            (1.0 + dynamic_exponent) * potential
            - field_dimension * self.field * self.derivative(potential, 1)
            + potential_source
        )
        inertia_rate = (
            temporal_source
            - eta_t * inertia
            - field_dimension * self.field * self.derivative(inertia, 1)
        )
        stiffness_rate = (
            spatial_source
            - eta_x * stiffness
            - field_dimension * self.field * self.derivative(stiffness, 1)
        )
        self.geometry.update(
            {
                "eta_t": float(eta_t),
                "eta_x": float(eta_x),
                "dynamic_exponent": float(dynamic_exponent),
                "field_dimension": float(field_dimension),
            }
        )
        return potential_rate, inertia_rate, stiffness_rate

    def potential_rate(self, potential, *, inertia=None, stiffness=None):
        """Return ``beta_U`` while projecting kinetic flow only at normalization."""
        potential = self.grid._field_values(potential, "potential")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self.grid._field_values(inertia, "inertia")
        stiffness = self.grid._field_values(stiffness, "stiffness")
        curvature = self.derivative(potential, 2)
        cubic = self.derivative(potential, 3)
        quartic = self.derivative(potential, 4)
        source = np.empty_like(self.field)
        for index in range(self.field.size):
            source[index] = self._point_sources(
                curvature[index],
                cubic[index],
                quartic[index],
                kinetic=False,
                inertia=inertia[index],
                stiffness=stiffness[index],
            )[0]
        center = self.normalize_index
        _, eta_t, eta_x = self._point_sources(
            curvature[center],
            cubic[center],
            quartic[center],
            kinetic=True,
            inertia=inertia[center],
            stiffness=stiffness[center],
        )
        eta_t *= self.kinetic_strength
        eta_x *= self.kinetic_strength
        dynamic_exponent = 1.0 + 0.5 * (eta_t - eta_x)
        field_dimension = 0.5 * (
            dynamic_exponent - 1.0 + eta_x
        )
        self.geometry = {
            "curvature": curvature,
            "cubic": cubic,
            "quartic": quartic,
            "potential_source": source,
            "eta_t": float(eta_t),
            "eta_x": float(eta_x),
            "dynamic_exponent": float(dynamic_exponent),
            "field_dimension": float(field_dimension),
        }
        return (
            (1.0 + dynamic_exponent) * potential
            - field_dimension * self.field * self.derivative(potential, 1)
            + source
        )

    def solve_fixed_point(
        self,
        initial,
        *,
        even=True,
        tolerance=1.0e-9,
        max_iterations=400,
    ):
        """Solve the full field-grid fixed-point equation in place."""
        from scipy.optimize import root

        initial = self.grid._field_values(initial, "initial")
        if even:
            if self.field.size % 2 != 1 or not np.allclose(
                self.field, -self.field[::-1]
            ):
                raise ValueError("even fixed points require an odd symmetric grid.")
            center = self.field.size // 2

            def expand(values):
                return np.concatenate((values[:0:-1], values))

            guess = initial[center:]

            def residual(values):
                try:
                    return self.potential_rate(expand(values))[center:]
                except ValueError:
                    return np.full_like(values, 1.0e6)

            solution = root(
                residual,
                guess,
                method="krylov",
                options={"fatol": tolerance, "maxiter": max_iterations},
            )
            potential = expand(solution.x)
        else:
            solution = root(
                self.potential_rate,
                initial,
                method="krylov",
                options={"fatol": tolerance, "maxiter": max_iterations},
            )
            potential = solution.x
        self.fixed_potential = np.asarray(potential)
        self.fixed_beta = self.potential_rate(self.fixed_potential)
        self.success = bool(
            solution.success
            or np.max(np.abs(self.fixed_beta)) <= 10.0 * tolerance
        )
        self.message = str(solution.message)
        return self

    def solve_coupled_fixed_point(
        self,
        initial,
        *,
        inertia=None,
        stiffness=None,
        tolerance=1.0e-8,
        max_iterations=500,
    ):
        """Solve the even full-grid fixed point for U, Zt, and Zx."""
        from scipy.optimize import root

        initial = self.grid._field_values(initial, "initial")
        if self.field.size % 2 != 1 or not np.allclose(
            self.field, -self.field[::-1]
        ):
            raise ValueError("coupled fixed points require an odd symmetric grid.")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self.grid._field_values(inertia, "inertia")
        stiffness = self.grid._field_values(stiffness, "stiffness")
        if np.any(inertia <= 0.0) or np.any(stiffness <= 0.0):
            raise ValueError("initial kinetic functions must be positive.")
        center = self.field.size // 2
        size = self.field.size - center

        def expand(values):
            return np.concatenate((values[:0:-1], values))

        guess = np.concatenate(
            (
                initial[center:],
                np.log(inertia[center:]),
                np.log(stiffness[center:]),
            )
        )

        def unpack(values):
            potential = expand(values[:size])
            temporal = expand(np.exp(values[size : 2 * size]))
            spatial = expand(np.exp(values[2 * size :]))
            return potential, temporal, spatial

        def residual(values):
            try:
                potential, temporal, spatial = unpack(values)
                rates = self.rates(
                    potential, inertia=temporal, stiffness=spatial
                )
                out = np.concatenate(tuple(rate[center:] for rate in rates))
                out[size] = temporal[center] - 1.0
                out[2 * size] = spatial[center] - 1.0
                return out
            except ValueError:
                return np.full_like(values, 1.0e6)

        initial_residual = residual(guess)
        if np.max(np.abs(initial_residual)) <= tolerance:
            potential, temporal, spatial = unpack(guess)
            self.fixed_potential = potential
            self.fixed_inertia = temporal
            self.fixed_stiffness = spatial
            self.fixed_beta = initial_residual
            self.success = True
            self.message = "initial coupled state satisfies the fixed-point equations"
            return self
        solution = root(
            residual,
            guess,
            method="krylov",
            options={"fatol": tolerance, "maxiter": max_iterations},
        )
        potential, temporal, spatial = unpack(solution.x)
        self.fixed_potential = potential
        self.fixed_inertia = temporal
        self.fixed_stiffness = spatial
        rates = self.rates(
            potential, inertia=temporal, stiffness=spatial
        )
        fixed_beta = np.concatenate(tuple(rate[center:] for rate in rates))
        fixed_beta[size] = temporal[center] - 1.0
        fixed_beta[2 * size] = spatial[center] - 1.0
        self.fixed_beta = fixed_beta
        self.success = bool(
            solution.success or np.max(np.abs(fixed_beta)) <= 10.0 * tolerance
        )
        self.message = str(solution.message)
        return self

    def solve_spectral_fixed_point(
        self,
        initial,
        *,
        inertia=None,
        stiffness=None,
        modes=5,
        tolerance=1.0e-7,
        max_evaluations=600,
    ):
        """Solve the normalized even U, Zt, Zx Chebyshev coefficient flow."""
        from scipy.optimize import root

        initial = self.grid._field_values(initial, "initial")
        if self.field.size % 2 != 1 or not np.allclose(
            self.field, -self.field[::-1]
        ):
            raise ValueError("spectral fixed points require an odd symmetric grid.")
        if inertia is None:
            inertia = np.ones_like(self.field)
        if stiffness is None:
            stiffness = np.ones_like(self.field)
        inertia = self.grid._field_values(inertia, "inertia")
        stiffness = self.grid._field_values(stiffness, "stiffness")
        modes = int(modes)
        if modes < 3 or 3 * modes > self.field.size:
            raise ValueError("modes must satisfy 3 <= 3*modes <= field.size.")
        coordinate = self.field / np.max(np.abs(self.field))
        degrees = 2 * np.arange(modes)

        def coefficients(values):
            fitted = np.polynomial.chebyshev.chebfit(
                coordinate, values, degrees[-1]
            )
            return fitted[degrees]
        center = self.field.size // 2
        shape_columns = []
        for degree in degrees[1:]:
            full = np.zeros(degree + 1)
            full[degree] = 1.0
            values = np.polynomial.chebyshev.chebval(coordinate, full)
            shape_columns.append(values - values[center])
        shape_basis = np.column_stack(shape_columns)
        shape_projector = np.linalg.pinv(shape_basis)
        block = modes - 1
        kinetic_scale = max(self.kinetic_strength, 1.0e-8)
        guess = np.concatenate(
            (
                coefficients(initial),
                shape_projector
                @ (np.log(inertia) - np.log(inertia[center]))
                / kinetic_scale,
                shape_projector
                @ (np.log(stiffness) - np.log(stiffness[center]))
                / kinetic_scale,
            )
        )

        def evaluate(values):
            full = np.zeros(degrees[-1] + 1)
            full[degrees] = values[:modes]
            potential = np.polynomial.chebyshev.chebval(coordinate, full)
            temporal = np.exp(
                kinetic_scale * shape_basis @ values[modes : modes + block]
            )
            spatial = np.exp(
                kinetic_scale * shape_basis @ values[modes + block :]
            )
            return potential, temporal, spatial

        def residual(values):
            try:
                potential, temporal, spatial = evaluate(values)
                potential_rate, temporal_rate, spatial_rate = self.rates(
                    potential, inertia=temporal, stiffness=spatial
                )
                return np.concatenate(
                    (
                        coefficients(potential_rate),
                        shape_projector
                        @ (temporal_rate / temporal)
                        / kinetic_scale,
                        shape_projector
                        @ (spatial_rate / spatial)
                        / kinetic_scale,
                    )
                )
            except ValueError:
                return np.full(modes + 2 * block, 1.0e3)

        solution = root(
            residual,
            guess,
            method="hybr",
            options={
                "xtol": tolerance,
                "maxfev": max_evaluations,
            },
        )
        potential, temporal, spatial = evaluate(solution.x)
        self.fixed_potential = potential
        self.fixed_inertia = temporal
        self.fixed_stiffness = spatial
        self.fixed_beta = residual(solution.x)
        self.success = bool(
            np.max(np.abs(self.fixed_beta)) <= 20.0 * tolerance
        )
        self.message = str(solution.message)
        self.spectral_coefficients = solution.x
        return self

    def solve_spectral_potential_fixed_point(
        self,
        initial,
        *,
        inertia=None,
        stiffness=None,
        modes=5,
        tolerance=1.0e-8,
        max_evaluations=600,
    ):
        """Solve the even projected potential flow at fixed kinetic functions."""
        from scipy.optimize import root

        initial = self.grid._field_values(initial, "initial")
        if self.field.size % 2 != 1 or not np.allclose(
            self.field, -self.field[::-1]
        ):
            raise ValueError("spectral fixed points require an odd symmetric grid.")
        inertia = (
            np.ones_like(self.field)
            if inertia is None
            else self.grid._field_values(inertia, "inertia")
        )
        stiffness = (
            np.ones_like(self.field)
            if stiffness is None
            else self.grid._field_values(stiffness, "stiffness")
        )
        modes = int(modes)
        if modes < 3 or modes > self.field.size // 2:
            raise ValueError("modes must lie between three and half the grid size.")
        coordinate = self.field / np.max(np.abs(self.field))
        degrees = 2 * np.arange(modes)
        fitted = np.polynomial.chebyshev.chebfit(
            coordinate, initial, degrees[-1]
        )
        guess = fitted[degrees]
        center = self.field.size // 2

        def evaluate(values):
            coefficients = np.zeros(degrees[-1] + 1)
            coefficients[degrees] = values
            return np.polynomial.chebyshev.chebval(coordinate, coefficients)

        def residual(values):
            try:
                rate = self.potential_rate(
                    evaluate(values), inertia=inertia, stiffness=stiffness
                )
                fitted = np.polynomial.chebyshev.chebfit(
                    coordinate, rate, degrees[-1]
                )
                return fitted[degrees]
            except ValueError:
                return np.full(modes, 1.0e3)

        solution = root(
            residual,
            guess,
            method="hybr",
            options={
                "xtol": tolerance,
                "maxfev": max_evaluations,
            },
        )
        self.fixed_potential = evaluate(solution.x)
        self.fixed_inertia = inertia.copy()
        self.fixed_stiffness = stiffness.copy()
        self.fixed_beta = residual(solution.x)
        self.success = bool(
            np.max(np.abs(self.fixed_beta)) <= 20.0 * tolerance
        )
        self.message = str(solution.message)
        self.spectral_coefficients = solution.x
        return self

    def continue_potential_modes(
        self,
        initial,
        mode_counts,
        *,
        inertia=None,
        stiffness=None,
        homotopy_steps=10,
        minimum_step=1.0e-3,
        minimum_interaction_fraction=0.2,
        tolerance=1.0e-8,
        max_evaluations=600,
    ):
        r"""Track the interacting potential fixed point into richer bases.

        New Chebyshev equations are activated continuously.  At homotopy
        strength zero, the added coefficients are pinned to their embedded
        lower-mode values; at strength one, all projected beta functions
        vanish.  Steps that lose the sign or a requested fraction of the
        reference quartic vertex are rejected and retried at half the step.
        """
        from scipy.optimize import root

        potential = self.grid._field_values(initial, "initial")
        inertia = (
            np.ones_like(self.field)
            if inertia is None
            else self.grid._field_values(inertia, "inertia")
        )
        stiffness = (
            np.ones_like(self.field)
            if stiffness is None
            else self.grid._field_values(stiffness, "stiffness")
        )
        mode_counts = np.asarray(mode_counts, dtype=int)
        if mode_counts.ndim != 1 or mode_counts.size == 0:
            raise ValueError("mode_counts must be a nonempty one-dimensional array.")
        if np.any(mode_counts < 3) or np.any(np.diff(mode_counts) <= 0):
            raise ValueError("mode_counts must be strictly increasing from three.")
        if mode_counts[-1] > self.field.size // 2:
            raise ValueError("mode counts cannot exceed half the field-grid size.")
        homotopy_steps = int(homotopy_steps)
        minimum_step = float(minimum_step)
        minimum_interaction_fraction = float(minimum_interaction_fraction)
        if homotopy_steps < 1 or not 0.0 < minimum_step <= 1.0:
            raise ValueError("homotopy_steps and minimum_step must be positive.")
        if not 0.0 <= minimum_interaction_fraction < 1.0:
            raise ValueError("minimum_interaction_fraction must lie in [0, 1).")

        def diagnostics(values, reference=None):
            center = self.normalize_index
            curvature = self.derivative(values, 2)
            quartic = float(self.derivative(values, 4)[center])
            overlap = 1.0
            if reference is not None:
                left = values - values[center]
                right = reference - reference[center]
                denominator = np.linalg.norm(left) * np.linalg.norm(right)
                overlap = (
                    float(np.dot(left, right) / denominator)
                    if denominator > 0.0
                    else 1.0
                )
            return {
                "mass2": float(curvature[center]),
                "quartic": quartic,
                "minimum_gap2": float(
                    np.min(self.kernel.cutoff**2 + curvature)
                ),
                "shape_overlap": overlap,
            }

        first_modes = int(mode_counts[0])
        self.solve_spectral_potential_fixed_point(
            potential,
            inertia=inertia,
            stiffness=stiffness,
            modes=first_modes,
            tolerance=tolerance,
            max_evaluations=max_evaluations,
        )
        potential = self.fixed_potential.copy()
        reference_quartic = diagnostics(potential)["quartic"]
        history = [
            {
                "modes": first_modes,
                "homotopy": 1.0,
                "step": 0.0,
                "accepted": bool(self.success),
                "success": bool(self.success),
                "message": self.message,
                "max_residual": float(np.max(np.abs(self.fixed_beta))),
                **diagnostics(potential),
            }
        ]
        if not self.success:
            self.mode_continuation = history
            return self

        previous_modes = first_modes
        for target_modes in mode_counts[1:]:
            target_modes = int(target_modes)
            coordinate = self.field / np.max(np.abs(self.field))
            degrees = 2 * np.arange(target_modes)
            fitted = np.polynomial.chebyshev.chebfit(
                coordinate, potential, degrees[-1]
            )
            current = fitted[degrees]
            anchor = current.copy()
            previous_potential = potential.copy()

            def evaluate(parameters):
                coefficients = np.zeros(degrees[-1] + 1)
                coefficients[degrees] = parameters
                return np.polynomial.chebyshev.chebval(
                    coordinate, coefficients
                )

            def full_residual(parameters):
                try:
                    rate = self.potential_rate(
                        evaluate(parameters),
                        inertia=inertia,
                        stiffness=stiffness,
                    )
                    coefficients = np.polynomial.chebyshev.chebfit(
                        coordinate, rate, degrees[-1]
                    )
                    return coefficients[degrees]
                except ValueError:
                    return np.full(target_modes, 1.0e3)

            def residual(parameters, strength):
                values = full_residual(parameters)
                values[previous_modes:] = (
                    strength * values[previous_modes:]
                    + (1.0 - strength)
                    * (parameters[previous_modes:] - anchor[previous_modes:])
                )
                return values

            strength = 0.0
            trial_step = 1.0 / homotopy_steps
            stage_success = True
            while strength < 1.0 - 1.0e-14:
                trial_strength = min(1.0, strength + trial_step)
                solution = root(
                    lambda parameters: residual(parameters, trial_strength),
                    current,
                    method="hybr",
                    options={
                        "xtol": tolerance,
                        "maxfev": max_evaluations,
                    },
                )
                trial_potential = evaluate(solution.x)
                point = diagnostics(trial_potential, previous_potential)
                homotopy_residual = residual(solution.x, trial_strength)
                residual_ok = bool(
                    np.max(np.abs(homotopy_residual)) <= 20.0 * tolerance
                )
                interaction_ok = bool(
                    reference_quartic == 0.0
                    or (
                        np.sign(point["quartic"])
                        == np.sign(reference_quartic)
                        and abs(point["quartic"])
                        >= minimum_interaction_fraction
                        * abs(reference_quartic)
                    )
                )
                accepted = residual_ok and interaction_ok
                history.append(
                    {
                        "modes": target_modes,
                        "homotopy": float(trial_strength),
                        "step": float(trial_step),
                        "accepted": accepted,
                        "success": bool(solution.success or residual_ok),
                        "message": str(solution.message),
                        "max_residual": float(
                            np.max(np.abs(homotopy_residual))
                        ),
                        "interaction_ok": interaction_ok,
                        **point,
                    }
                )
                if accepted:
                    current = solution.x
                    potential = trial_potential
                    strength = trial_strength
                    previous_potential = potential.copy()
                    trial_step = min(1.0 - strength, 1.5 * trial_step)
                    continue
                trial_step *= 0.5
                if trial_step < minimum_step:
                    stage_success = False
                    break

            self.fixed_potential = potential.copy()
            self.fixed_inertia = inertia.copy()
            self.fixed_stiffness = stiffness.copy()
            self.spectral_coefficients = current.copy()
            self.fixed_beta = full_residual(current)
            self.success = bool(
                stage_success
                and strength >= 1.0 - 1.0e-14
                and np.max(np.abs(self.fixed_beta)) <= 20.0 * tolerance
            )
            self.message = (
                "interacting mode continuation converged"
                if self.success
                else "interacting mode continuation stopped before the full equations"
            )
            if not self.success:
                break
            previous_modes = target_modes

        self.mode_continuation = history
        return self

    def continue_feshbach_potential_fixed_point(
        self,
        initial,
        strengths,
        *,
        inertia=None,
        stiffness=None,
        modes=3,
        minimum_interaction_fraction=0.2,
        tolerance=1.0e-8,
        max_evaluations=600,
    ):
        """Track the interacting potential branch through Feshbach strength."""
        potential = self.grid._field_values(initial, "initial")
        temporal = (
            np.ones_like(self.field)
            if inertia is None
            else self.grid._field_values(inertia, "inertia")
        )
        spatial = (
            np.ones_like(self.field)
            if stiffness is None
            else self.grid._field_values(stiffness, "stiffness")
        )
        strengths = np.asarray(strengths, dtype=float)
        if strengths.ndim != 1 or strengths.size == 0:
            raise ValueError("strengths must be a nonempty one-dimensional array.")
        if np.any((strengths < 0.0) | (strengths > 1.0)):
            raise ValueError("Feshbach strengths must lie between zero and one.")
        differences = np.diff(strengths)
        if differences.size and not (
            np.all(differences > 0.0) or np.all(differences < 0.0)
        ):
            raise ValueError("Feshbach strengths must be strictly monotone.")
        reference_quartic = None
        history = []
        last_accepted = (potential.copy(), temporal.copy(), spatial.copy())
        for strength in strengths:
            self.kernel.set_feshbach_strength(float(strength))
            self.solve_spectral_potential_fixed_point(
                potential,
                inertia=temporal,
                stiffness=spatial,
                modes=modes,
                tolerance=tolerance,
                max_evaluations=max_evaluations,
            )
            center = self.normalize_index
            quartic = float(
                self.derivative(self.fixed_potential, 4)[center]
            )
            if reference_quartic is None:
                reference_quartic = quartic
            interaction_ok = bool(
                abs(reference_quartic) <= 100.0 * tolerance
                or (
                    np.sign(quartic) == np.sign(reference_quartic)
                    and abs(quartic)
                    >= minimum_interaction_fraction
                    * abs(reference_quartic)
                )
            )
            accepted = bool(self.success and interaction_ok)
            history.append(
                {
                    "strength": float(strength),
                    "success": accepted,
                    "max_residual": float(np.max(np.abs(self.fixed_beta))),
                    "mass2": float(
                        self.derivative(self.fixed_potential, 2)[center]
                    ),
                    "quartic": quartic,
                    "interaction_ok": interaction_ok,
                    "potential": self.fixed_potential.copy(),
                }
            )
            if not accepted:
                self.success = False
                self.message = "Feshbach continuation left the interacting branch"
                potential, temporal, spatial = (
                    values.copy() for values in last_accepted
                )
                self.fixed_potential = potential.copy()
                self.fixed_inertia = temporal.copy()
                self.fixed_stiffness = spatial.copy()
                break
            potential = self.fixed_potential.copy()
            temporal = self.fixed_inertia.copy()
            spatial = self.fixed_stiffness.copy()
            last_accepted = (potential, temporal, spatial)
        self.feshbach_continuation = history
        return self

    def continue_kinetic_fixed_point(
        self,
        initial,
        strengths,
        *,
        inertia=None,
        stiffness=None,
        modes=5,
        preserve_interacting_branch=True,
        minimum_interaction_fraction=0.2,
        homotopy_steps=8,
        minimum_step=1.0e-3,
        tolerance=1.0e-7,
        max_evaluations=600,
    ):
        """Track a coupled fixed point while turning on kinetic feedback.

        A supplied zero-strength fixed point is reused rather than solved
        again.  This matters after mode homotopy because an unconstrained
        solve at the same endpoint can jump back to the Gaussian branch.
        """
        from scipy.optimize import root

        potential = self.grid._field_values(initial, "initial")
        temporal = (
            np.ones_like(self.field)
            if inertia is None
            else self.grid._field_values(inertia, "inertia")
        )
        spatial = (
            np.ones_like(self.field)
            if stiffness is None
            else self.grid._field_values(stiffness, "stiffness")
        )
        strengths = np.asarray(strengths, dtype=float)
        if (
            strengths.ndim != 1
            or strengths.size == 0
            or strengths[0] != 0.0
            or np.any(np.diff(strengths) <= 0.0)
            or strengths[-1] > 1.0
        ):
            raise ValueError(
                "strengths must increase strictly from zero to at most one."
            )
        modes = int(modes)
        if modes < 3 or 3 * modes > self.field.size:
            raise ValueError("modes must satisfy 3 <= 3*modes <= field.size.")
        coordinate = self.field / np.max(np.abs(self.field))
        degrees = 2 * np.arange(modes)
        center = self.normalize_index
        shape_columns = []
        for degree in degrees[1:]:
            coefficients = np.zeros(degree + 1)
            coefficients[degree] = 1.0
            values = np.polynomial.chebyshev.chebval(coordinate, coefficients)
            shape_columns.append(values - values[center])
        shape_basis = np.column_stack(shape_columns)
        shape_projector = np.linalg.pinv(shape_basis)
        block = modes - 1

        def coefficients(values):
            fitted = np.polynomial.chebyshev.chebfit(
                coordinate, values, degrees[-1]
            )
            return fitted[degrees]

        parameters = np.concatenate(
            (
                coefficients(potential),
                shape_projector
                @ (np.log(temporal) - np.log(temporal[center])),
                shape_projector
                @ (np.log(spatial) - np.log(spatial[center])),
            )
        )

        def evaluate(values):
            full = np.zeros(degrees[-1] + 1)
            full[degrees] = values[:modes]
            trial_potential = np.polynomial.chebyshev.chebval(
                coordinate, full
            )
            trial_temporal = np.exp(
                shape_basis @ values[modes : modes + block]
            )
            trial_spatial = np.exp(shape_basis @ values[modes + block :])
            return trial_potential, trial_temporal, trial_spatial

        def physical_residual(values, strength):
            self.set_kinetic_strength(strength)
            try:
                trial_potential, trial_temporal, trial_spatial = evaluate(values)
                rates = self.rates(
                    trial_potential,
                    inertia=trial_temporal,
                    stiffness=trial_spatial,
                )
                return np.concatenate(
                    (
                        coefficients(rates[0]),
                        shape_projector @ (rates[1] / trial_temporal),
                        shape_projector @ (rates[2] / trial_spatial),
                    )
                )
            except ValueError:
                return np.full(modes + 2 * block, 1.0e3)

        zero_anchor = parameters.copy()

        def zero_residual(values):
            residual = physical_residual(values, 0.0)
            residual[modes:] = values[modes:] - zero_anchor[modes:]
            return residual

        history = []
        homotopy_history = []
        reference_quartic = float(
            self.derivative(potential, 4)[self.normalize_index]
        )
        initial_residual = zero_residual(parameters)
        if np.max(np.abs(initial_residual)) > 20.0 * tolerance:
            solution = root(
                zero_residual,
                parameters,
                method="hybr",
                options={"xtol": tolerance, "maxfev": max_evaluations},
            )
            parameters = solution.x
            initial_residual = zero_residual(parameters)
        potential, temporal, spatial = evaluate(parameters)
        self.set_kinetic_strength(0.0)
        self.potential_rate(potential, inertia=temporal, stiffness=spatial)
        initial_success = bool(
            np.max(np.abs(initial_residual)) <= 20.0 * tolerance
        )
        history.append(
            {
                "strength": 0.0,
                "success": initial_success,
                "message": "initialized kinetic homotopy",
                "max_residual": float(np.max(np.abs(initial_residual))),
                "potential": potential.copy(),
                "inertia": temporal.copy(),
                "stiffness": spatial.copy(),
                "quartic": reference_quartic,
                "interaction_ok": True,
                "eta_t": float(self.geometry["eta_t"]),
                "eta_x": float(self.geometry["eta_x"]),
                "dynamic_exponent": float(self.geometry["dynamic_exponent"]),
            }
        )
        previous_strength = 0.0
        stage_success = initial_success
        for target_strength in strengths[1:]:
            target_strength = float(target_strength)
            start_parameters = parameters.copy()

            start_potential, start_temporal, start_spatial = evaluate(
                start_parameters
            )
            self.set_kinetic_strength(target_strength)
            self.solve_spectral_fixed_point(
                start_potential,
                inertia=start_temporal,
                stiffness=start_spatial,
                modes=modes,
                tolerance=tolerance,
                max_evaluations=max_evaluations,
            )
            direct_quartic = float(
                self.derivative(self.fixed_potential, 4)[center]
            )
            direct_interaction_ok = bool(
                not preserve_interacting_branch
                or reference_quartic == 0.0
                or (
                    np.sign(direct_quartic) == np.sign(reference_quartic)
                    and abs(direct_quartic)
                    >= minimum_interaction_fraction
                    * abs(reference_quartic)
                )
            )
            direct_ok = bool(self.success and direct_interaction_ok)
            homotopy_history.append(
                {
                    "start_strength": float(previous_strength),
                    "target_strength": target_strength,
                    "fraction": 1.0,
                    "step": 1.0,
                    "accepted": direct_ok,
                    "method": "direct",
                    "max_residual": float(np.max(np.abs(self.fixed_beta))),
                    "quartic": direct_quartic,
                    "interaction_ok": direct_interaction_ok,
                }
            )
            if direct_ok:
                potential = self.fixed_potential.copy()
                temporal = self.fixed_inertia.copy()
                spatial = self.fixed_stiffness.copy()
                parameters = np.concatenate(
                    (
                        coefficients(potential),
                        shape_projector
                        @ (np.log(temporal) - np.log(temporal[center])),
                        shape_projector
                        @ (np.log(spatial) - np.log(spatial[center])),
                    )
                )
                endpoint_residual = physical_residual(
                    parameters, target_strength
                )
                previous_strength = target_strength
                history.append(
                    {
                        "strength": target_strength,
                        "success": True,
                        "message": "direct branch-preserving step converged",
                        "max_residual": float(
                            np.max(np.abs(endpoint_residual))
                        ),
                        "potential": potential.copy(),
                        "inertia": temporal.copy(),
                        "stiffness": spatial.copy(),
                        "quartic": direct_quartic,
                        "interaction_ok": True,
                        "eta_t": float(self.geometry["eta_t"]),
                        "eta_x": float(self.geometry["eta_x"]),
                        "dynamic_exponent": float(
                            self.geometry["dynamic_exponent"]
                        ),
                    }
                )
                continue
            parameters = start_parameters
            self.set_kinetic_strength(previous_strength)

            def left_residual(values):
                if previous_strength == 0.0:
                    return zero_residual(values)
                return physical_residual(values, previous_strength)

            def homotopy_residual(values, fraction):
                return (
                    (1.0 - fraction) * left_residual(values)
                    + fraction * physical_residual(values, target_strength)
                )

            fraction = 0.0
            trial_step = 1.0 / int(homotopy_steps)
            while fraction < 1.0 - 1.0e-14:
                trial_fraction = min(1.0, fraction + trial_step)
                solution = root(
                    lambda values: homotopy_residual(values, trial_fraction),
                    parameters,
                    method="hybr",
                    options={"xtol": tolerance, "maxfev": max_evaluations},
                )
                trial_potential, trial_temporal, trial_spatial = evaluate(
                    solution.x
                )
                quartic = float(
                    self.derivative(trial_potential, 4)[center]
                )
                residual = homotopy_residual(solution.x, trial_fraction)
                residual_ok = bool(
                    np.max(np.abs(residual)) <= 20.0 * tolerance
                )
                interaction_ok = bool(
                    not preserve_interacting_branch
                    or reference_quartic == 0.0
                    or (
                        np.sign(quartic) == np.sign(reference_quartic)
                        and abs(quartic)
                        >= minimum_interaction_fraction
                        * abs(reference_quartic)
                    )
                )
                accepted = residual_ok and interaction_ok
                homotopy_history.append(
                    {
                        "start_strength": float(previous_strength),
                        "target_strength": target_strength,
                        "fraction": float(trial_fraction),
                        "step": float(trial_step),
                        "accepted": accepted,
                        "method": "homotopy",
                        "max_residual": float(np.max(np.abs(residual))),
                        "quartic": quartic,
                        "interaction_ok": interaction_ok,
                    }
                )
                if accepted:
                    parameters = solution.x
                    fraction = trial_fraction
                    trial_step = min(1.0 - fraction, 1.5 * trial_step)
                    continue
                trial_step *= 0.5
                if trial_step < minimum_step:
                    stage_success = False
                    parameters = start_parameters
                    break
            if not stage_success:
                break
            previous_strength = target_strength
            potential, temporal, spatial = evaluate(parameters)
            endpoint_residual = physical_residual(parameters, target_strength)
            quartic = float(self.derivative(potential, 4)[center])
            history.append(
                {
                    "strength": target_strength,
                    "success": True,
                    "message": "kinetic homotopy converged",
                    "max_residual": float(
                        np.max(np.abs(endpoint_residual))
                    ),
                    "potential": potential.copy(),
                    "inertia": temporal.copy(),
                    "stiffness": spatial.copy(),
                    "quartic": quartic,
                    "interaction_ok": True,
                    "eta_t": float(self.geometry["eta_t"]),
                    "eta_x": float(self.geometry["eta_x"]),
                    "dynamic_exponent": float(
                        self.geometry["dynamic_exponent"]
                    ),
                }
            )
        potential, temporal, spatial = evaluate(parameters)
        self.fixed_potential = potential
        self.fixed_inertia = temporal
        self.fixed_stiffness = spatial
        self.fixed_beta = (
            physical_residual(parameters, previous_strength)
            if previous_strength > 0.0
            else zero_residual(parameters)
        )
        self.spectral_coefficients = parameters
        self.success = bool(
            stage_success
            and previous_strength == strengths[-1]
            and np.max(np.abs(self.fixed_beta)) <= 20.0 * tolerance
        )
        self.message = (
            "kinetic homotopy converged"
            if self.success
            else "kinetic homotopy stopped before the requested endpoint"
        )
        self.kinetic_continuation = history
        self.kinetic_homotopy = homotopy_history
        return self

    def continue_coupled_modes(
        self,
        initial,
        mode_counts,
        *,
        inertia=None,
        stiffness=None,
        homotopy_steps=10,
        minimum_step=1.0e-3,
        pseudo_arclength_steps=80,
        minimum_interaction_fraction=0.2,
        maximum_interaction_factor=10.0,
        tolerance=1.0e-7,
        max_evaluations=600,
    ):
        """Continue an interacting coupled fixed point to higher field modes."""
        from scipy.optimize import root

        potential = self.grid._field_values(initial, "initial")
        temporal = (
            np.ones_like(self.field)
            if inertia is None
            else self.grid._field_values(inertia, "inertia")
        )
        spatial = (
            np.ones_like(self.field)
            if stiffness is None
            else self.grid._field_values(stiffness, "stiffness")
        )
        mode_counts = np.asarray(mode_counts, dtype=int)
        maximum_interaction_factor = float(maximum_interaction_factor)
        if maximum_interaction_factor <= 1.0:
            raise ValueError("maximum_interaction_factor must exceed one.")
        if (
            mode_counts.ndim != 1
            or mode_counts.size == 0
            or np.any(mode_counts < 3)
            or np.any(np.diff(mode_counts) <= 0)
            or 3 * mode_counts[-1] > self.field.size
        ):
            raise ValueError(
                "mode_counts must increase strictly and satisfy 3*modes <= field.size."
            )
        center = self.normalize_index
        reference_quartic = float(self.derivative(potential, 4)[center])
        history = []
        previous_modes = int(mode_counts[0])

        def setup(target_modes, values, inertia_values, stiffness_values):
            coordinate = self.field / np.max(np.abs(self.field))
            degrees = 2 * np.arange(target_modes)
            columns = []
            for degree in degrees[1:]:
                coefficients = np.zeros(degree + 1)
                coefficients[degree] = 1.0
                column = np.polynomial.chebyshev.chebval(
                    coordinate, coefficients
                )
                columns.append(column - column[center])
            basis = np.column_stack(columns)
            projector = np.linalg.pinv(basis)
            block = target_modes - 1

            def potential_coefficients(field_values):
                fitted = np.polynomial.chebyshev.chebfit(
                    coordinate, field_values, degrees[-1]
                )
                return fitted[degrees]

            parameters = np.concatenate(
                (
                    potential_coefficients(values),
                    projector
                    @ (
                        np.log(inertia_values)
                        - np.log(inertia_values[center])
                    ),
                    projector
                    @ (
                        np.log(stiffness_values)
                        - np.log(stiffness_values[center])
                    ),
                )
            )

            def evaluate(parameters):
                coefficients = np.zeros(degrees[-1] + 1)
                coefficients[degrees] = parameters[:target_modes]
                trial_potential = np.polynomial.chebyshev.chebval(
                    coordinate, coefficients
                )
                trial_temporal = np.exp(
                    basis
                    @ parameters[target_modes : target_modes + block]
                )
                trial_spatial = np.exp(
                    basis @ parameters[target_modes + block :]
                )
                return trial_potential, trial_temporal, trial_spatial

            def residual(parameters):
                try:
                    trial_potential, trial_temporal, trial_spatial = evaluate(
                        parameters
                    )
                    rates = self.rates(
                        trial_potential,
                        inertia=trial_temporal,
                        stiffness=trial_spatial,
                    )
                    return np.concatenate(
                        (
                            potential_coefficients(rates[0]),
                            projector @ (rates[1] / trial_temporal),
                            projector @ (rates[2] / trial_spatial),
                        )
                    )
                except ValueError:
                    return np.full(target_modes + 2 * block, 1.0e3)

            return parameters, evaluate, residual

        parameters, evaluate, full_residual = setup(
            previous_modes, potential, temporal, spatial
        )
        initial_residual = full_residual(parameters)
        if np.max(np.abs(initial_residual)) > 20.0 * tolerance:
            solution = root(
                full_residual,
                parameters,
                method="hybr",
                options={"xtol": tolerance, "maxfev": max_evaluations},
            )
            parameters = solution.x
            initial_residual = full_residual(parameters)
        potential, temporal, spatial = evaluate(parameters)
        quartic = float(self.derivative(potential, 4)[center])
        initial_ok = bool(
            np.max(np.abs(initial_residual)) <= 20.0 * tolerance
            and (
                reference_quartic == 0.0
                or (
                    np.sign(quartic) == np.sign(reference_quartic)
                    and abs(quartic)
                    >= minimum_interaction_fraction
                    * abs(reference_quartic)
                    and abs(quartic)
                    <= maximum_interaction_factor * abs(reference_quartic)
                )
            )
        )
        history.append(
            {
                "modes": previous_modes,
                "homotopy": 1.0,
                "accepted": initial_ok,
                "max_residual": float(np.max(np.abs(initial_residual))),
                "quartic": quartic,
            }
        )
        stage_success = initial_ok

        for target_modes in mode_counts[1:]:
            if not stage_success:
                break
            target_modes = int(target_modes)
            parameters, evaluate, full_residual = setup(
                target_modes, potential, temporal, spatial
            )
            anchor = parameters.copy()
            target_block = target_modes - 1
            old_block = previous_modes - 1
            new_indices = np.concatenate(
                (
                    np.arange(previous_modes, target_modes),
                    np.arange(
                        target_modes + old_block,
                        target_modes + target_block,
                    ),
                    np.arange(
                        target_modes + target_block + old_block,
                        target_modes + 2 * target_block,
                    ),
                )
            )

            def homotopy_residual(values, fraction):
                residual = full_residual(values)
                residual[new_indices] = (
                    fraction * residual[new_indices]
                    + (1.0 - fraction)
                    * (values[new_indices] - anchor[new_indices])
                )
                return residual

            fraction = 0.0
            trial_step = 1.0 / int(homotopy_steps)
            accepted_states = [np.concatenate((parameters, [fraction]))]
            while fraction < 1.0 - 1.0e-14:
                trial_fraction = min(1.0, fraction + trial_step)
                solution = root(
                    lambda values: homotopy_residual(values, trial_fraction),
                    parameters,
                    method="hybr",
                    options={"xtol": tolerance, "maxfev": max_evaluations},
                )
                trial_potential, trial_temporal, trial_spatial = evaluate(
                    solution.x
                )
                residual = homotopy_residual(
                    solution.x, trial_fraction
                )
                quartic = float(
                    self.derivative(trial_potential, 4)[center]
                )
                residual_ok = bool(
                    np.max(np.abs(residual)) <= 20.0 * tolerance
                )
                interaction_ok = bool(
                    reference_quartic == 0.0
                    or (
                        np.sign(quartic) == np.sign(reference_quartic)
                        and abs(quartic)
                        >= minimum_interaction_fraction
                        * abs(reference_quartic)
                        and abs(quartic)
                        <= maximum_interaction_factor
                        * abs(reference_quartic)
                    )
                )
                accepted = residual_ok and interaction_ok
                history.append(
                    {
                        "modes": target_modes,
                        "homotopy": float(trial_fraction),
                        "step": float(trial_step),
                        "accepted": accepted,
                        "max_residual": float(np.max(np.abs(residual))),
                        "quartic": quartic,
                        "interaction_ok": interaction_ok,
                    }
                )
                if accepted:
                    parameters = solution.x
                    potential = trial_potential
                    temporal = trial_temporal
                    spatial = trial_spatial
                    fraction = trial_fraction
                    accepted_states.append(
                        np.concatenate((parameters, [fraction]))
                    )
                    trial_step = min(1.0 - fraction, 1.5 * trial_step)
                    continue
                trial_step *= 0.5
                if trial_step < minimum_step:
                    stage_success = False
                    if len(accepted_states) >= 2:
                        previous_state = accepted_states[-2]
                        state = accepted_states[-1]
                        tangent = state - previous_state
                        tangent /= np.linalg.norm(tangent)
                        arc_step = max(
                            5.0 * minimum_step,
                            2.0 * np.linalg.norm(state - previous_state),
                        )
                        for _ in range(int(pseudo_arclength_steps)):
                            prediction = state + arc_step * tangent

                            def augmented_residual(extended):
                                return np.concatenate(
                                    (
                                        homotopy_residual(
                                            extended[:-1], extended[-1]
                                        ),
                                        [
                                            np.dot(
                                                extended - prediction,
                                                tangent,
                                            )
                                        ],
                                    )
                                )

                            correction = root(
                                augmented_residual,
                                prediction,
                                method="hybr",
                                options={
                                    "xtol": tolerance,
                                    "maxfev": max_evaluations,
                                },
                            )
                            corrected = correction.x
                            corrected_residual = augmented_residual(corrected)
                            trial_potential, trial_temporal, trial_spatial = (
                                evaluate(corrected[:-1])
                            )
                            quartic = float(
                                self.derivative(trial_potential, 4)[center]
                            )
                            residual_ok = bool(
                                np.max(np.abs(corrected_residual))
                                <= 100.0 * tolerance
                            )
                            interaction_ok = bool(
                                reference_quartic == 0.0
                                or (
                                    np.sign(quartic)
                                    == np.sign(reference_quartic)
                                    and abs(quartic)
                                    >= minimum_interaction_fraction
                                    * abs(reference_quartic)
                                    and abs(quartic)
                                    <= maximum_interaction_factor
                                    * abs(reference_quartic)
                                )
                            )
                            accepted = residual_ok and interaction_ok
                            history.append(
                                {
                                    "modes": target_modes,
                                    "homotopy": float(corrected[-1]),
                                    "step": float(arc_step),
                                    "accepted": accepted,
                                    "method": "pseudo-arclength",
                                    "max_residual": float(
                                        np.max(np.abs(corrected_residual))
                                    ),
                                    "quartic": quartic,
                                    "interaction_ok": interaction_ok,
                                }
                            )
                            if not accepted:
                                arc_step *= 0.5
                                if arc_step < minimum_step:
                                    break
                                continue
                            new_tangent = corrected - state
                            new_tangent /= np.linalg.norm(new_tangent)
                            if np.dot(new_tangent, tangent) < 0.0:
                                new_tangent *= -1.0
                            previous_state = state
                            state = corrected
                            tangent = new_tangent
                            parameters = state[:-1]
                            fraction = float(state[-1])
                            potential = trial_potential
                            temporal = trial_temporal
                            spatial = trial_spatial
                            arc_step = min(1.5 * arc_step, 0.25)

                            if fraction > 0.75:
                                endpoint = root(
                                    full_residual,
                                    parameters,
                                    method="hybr",
                                    options={
                                        "xtol": tolerance,
                                        "maxfev": max_evaluations,
                                    },
                                )
                                endpoint_potential, endpoint_temporal, endpoint_spatial = evaluate(
                                    endpoint.x
                                )
                                endpoint_quartic = float(
                                    self.derivative(endpoint_potential, 4)[
                                        center
                                    ]
                                )
                                endpoint_residual = full_residual(endpoint.x)
                                endpoint_ok = bool(
                                    np.max(np.abs(endpoint_residual))
                                    <= 20.0 * tolerance
                                    and np.sign(endpoint_quartic)
                                    == np.sign(reference_quartic)
                                    and abs(endpoint_quartic)
                                    >= minimum_interaction_fraction
                                    * abs(reference_quartic)
                                    and abs(endpoint_quartic)
                                    <= maximum_interaction_factor
                                    * abs(reference_quartic)
                                )
                                history.append(
                                    {
                                        "modes": target_modes,
                                        "homotopy": 1.0,
                                        "step": 0.0,
                                        "accepted": endpoint_ok,
                                        "method": "endpoint-corrector",
                                        "max_residual": float(
                                            np.max(np.abs(endpoint_residual))
                                        ),
                                        "quartic": endpoint_quartic,
                                        "interaction_ok": endpoint_ok,
                                    }
                                )
                                if endpoint_ok:
                                    parameters = endpoint.x
                                    potential = endpoint_potential
                                    temporal = endpoint_temporal
                                    spatial = endpoint_spatial
                                    fraction = 1.0
                                    stage_success = True
                                    break
                        if fraction >= 1.0 - 1.0e-14:
                            stage_success = True
                    break
            if stage_success:
                previous_modes = target_modes

        self.fixed_potential = potential.copy()
        self.fixed_inertia = temporal.copy()
        self.fixed_stiffness = spatial.copy()
        self.spectral_coefficients = parameters.copy()
        self.fixed_beta = full_residual(parameters)
        self.success = bool(
            stage_success
            and previous_modes == mode_counts[-1]
            and np.max(np.abs(self.fixed_beta)) <= 20.0 * tolerance
        )
        self.message = (
            "coupled mode continuation converged"
            if self.success
            else "coupled mode continuation stopped before the requested endpoint"
        )
        self.coupled_mode_continuation = history
        return self

    def probe_coupled_mode_extension(
        self,
        initial,
        source_modes,
        target_modes,
        *,
        inertia=None,
        stiffness=None,
        initial_step=0.05,
        minimum_step=2.0e-3,
        minimum_interaction_fraction=0.2,
        fold_tolerance=1.0e-4,
        tolerance=2.0e-7,
        max_evaluations=300,
    ):
        """Audit whether richer coupled equations connect to a fixed point.

        The newly added potential and metric equations are activated by a
        homotopy.  A scaled trust-region corrector is used because the three
        blocks have very different numerical scales.  Failure is reported
        together with the smallest Jacobian singular value and its block
        composition; an unreached endpoint never overwrites the input fixed
        point.
        """
        from scipy.optimize import least_squares

        source_modes = int(source_modes)
        target_modes = int(target_modes)
        if (
            source_modes < 3
            or target_modes <= source_modes
            or 3 * target_modes > self.field.size
        ):
            raise ValueError(
                "require 3 <= source_modes < target_modes and "
                "3*target_modes <= field.size"
            )
        initial_step = float(initial_step)
        minimum_step = float(minimum_step)
        if not 0.0 < minimum_step <= initial_step <= 1.0:
            raise ValueError("require 0 < minimum_step <= initial_step <= 1.")

        potential = self.grid._field_values(initial, "initial")
        temporal = (
            np.ones_like(self.field)
            if inertia is None
            else self.grid._field_values(inertia, "inertia")
        )
        spatial = (
            np.ones_like(self.field)
            if stiffness is None
            else self.grid._field_values(stiffness, "stiffness")
        )
        saved = (potential.copy(), temporal.copy(), spatial.copy())
        coordinate = self.field / np.max(np.abs(self.field))
        center = self.normalize_index
        degrees = 2 * np.arange(target_modes)
        columns = []
        for degree in degrees[1:]:
            coefficients = np.zeros(degree + 1)
            coefficients[degree] = 1.0
            values = np.polynomial.chebyshev.chebval(
                coordinate, coefficients
            )
            columns.append(values - values[center])
        basis = np.column_stack(columns)
        projector = np.linalg.pinv(basis)
        block = target_modes - 1

        def potential_coefficients(values):
            fitted = np.polynomial.chebyshev.chebfit(
                coordinate, values, degrees[-1]
            )
            return fitted[degrees]

        def evaluate(parameters):
            coefficients = np.zeros(degrees[-1] + 1)
            coefficients[degrees] = parameters[:target_modes]
            trial_potential = np.polynomial.chebyshev.chebval(
                coordinate, coefficients
            )
            trial_temporal = np.exp(
                basis
                @ parameters[target_modes : target_modes + block]
            )
            trial_spatial = np.exp(
                basis @ parameters[target_modes + block :]
            )
            return trial_potential, trial_temporal, trial_spatial

        parameters = np.concatenate(
            (
                potential_coefficients(potential),
                projector
                @ (np.log(temporal) - np.log(temporal[center])),
                projector
                @ (np.log(spatial) - np.log(spatial[center])),
            )
        )
        anchor = parameters.copy()
        source_block = source_modes - 1
        new_indices = np.concatenate(
            (
                np.arange(source_modes, target_modes),
                np.arange(
                    target_modes + source_block,
                    target_modes + block,
                ),
                np.arange(
                    target_modes + block + source_block,
                    target_modes + 2 * block,
                ),
            )
        )

        def full_residual(values):
            try:
                trial_potential, trial_temporal, trial_spatial = evaluate(
                    values
                )
                rates = self.rates(
                    trial_potential,
                    inertia=trial_temporal,
                    stiffness=trial_spatial,
                )
                return np.concatenate(
                    (
                        potential_coefficients(rates[0]),
                        projector @ (rates[1] / trial_temporal),
                        projector @ (rates[2] / trial_spatial),
                    )
                )
            except ValueError:
                return np.full(target_modes + 2 * block, 1.0e3)

        def homotopy_residual(values, strength):
            residual = full_residual(values)
            residual[new_indices] = (
                strength * residual[new_indices]
                + (1.0 - strength)
                * (values[new_indices] - anchor[new_indices])
            )
            return residual

        reference_quartic = float(
            self.derivative(potential, 4)[center]
        )
        strength = 0.0
        trial_step = initial_step
        history = []
        last_result = None
        last_accepted = parameters.copy()
        while strength < 1.0 - 1.0e-14:
            trial_strength = min(1.0, strength + trial_step)
            result = least_squares(
                lambda values: homotopy_residual(values, trial_strength),
                parameters,
                x_scale="jac",
                xtol=max(0.1 * tolerance, 1.0e-12),
                ftol=max(0.1 * tolerance, 1.0e-12),
                gtol=max(0.1 * tolerance, 1.0e-12),
                max_nfev=int(max_evaluations),
            )
            last_result = result
            residual = homotopy_residual(result.x, trial_strength)
            singular_values = np.linalg.svd(
                result.jac, compute_uv=False
            )
            trial_potential, trial_temporal, trial_spatial = evaluate(
                result.x
            )
            quartic = float(
                self.derivative(trial_potential, 4)[center]
            )
            residual_ok = bool(
                np.max(np.abs(residual)) <= 20.0 * tolerance
            )
            interaction_ok = bool(
                abs(reference_quartic) <= 100.0 * tolerance
                or (
                    np.sign(quartic) == np.sign(reference_quartic)
                    and abs(quartic)
                    >= minimum_interaction_fraction
                    * abs(reference_quartic)
                )
            )
            accepted = residual_ok and interaction_ok
            history.append(
                {
                    "strength": float(trial_strength),
                    "step": float(trial_step),
                    "accepted": accepted,
                    "max_residual": float(np.max(np.abs(residual))),
                    "quartic": quartic,
                    "interaction_ok": interaction_ok,
                    "smallest_jacobian_singular_value": float(
                        singular_values[-1]
                    ),
                }
            )
            if accepted:
                parameters = result.x
                last_accepted = parameters.copy()
                strength = trial_strength
                trial_step = min(1.0 - strength, 1.25 * trial_step)
                continue
            trial_step *= 0.5
            if trial_step < minimum_step:
                break

        reached = bool(strength >= 1.0 - 1.0e-14)
        singular_direction = None
        singular_fractions = None
        smallest_singular = float("nan")
        if last_result is not None:
            _, singular_values, right = np.linalg.svd(
                last_result.jac, full_matrices=False
            )
            smallest_singular = float(singular_values[-1])
            singular_direction = right[-1]
            weights = np.array(
                [
                    np.linalg.norm(
                        singular_direction[:target_modes]
                    ) ** 2,
                    np.linalg.norm(
                        singular_direction[
                            target_modes : target_modes + block
                        ]
                    ) ** 2,
                    np.linalg.norm(
                        singular_direction[target_modes + block :]
                    ) ** 2,
                ]
            )
            weights /= np.sum(weights)
            singular_fractions = {
                "potential": float(weights[0]),
                "temporal_metric": float(weights[1]),
                "spatial_metric": float(weights[2]),
            }
        near_fold = bool(
            not reached
            and np.isfinite(smallest_singular)
            and smallest_singular <= float(fold_tolerance)
        )
        self.mode_extension_probe = history
        self.mode_extension_diagnostics = {
            "source_modes": source_modes,
            "target_modes": target_modes,
            "endpoint_reached": reached,
            "maximum_activation": float(strength),
            "smallest_jacobian_singular_value": smallest_singular,
            "near_singular_fold": near_fold,
            "singular_direction_fractions": singular_fractions,
            "termination_reason": (
                "full higher-mode fixed point reached"
                if reached
                else (
                    "near-singular fold before the interacting endpoint"
                    if near_fold
                    else "interacting endpoint not reached"
                )
            ),
        }
        if reached:
            potential, temporal, spatial = evaluate(last_accepted)
            self.fixed_potential = potential
            self.fixed_inertia = temporal
            self.fixed_stiffness = spatial
            self.fixed_beta = full_residual(last_accepted)
            self.spectral_coefficients = last_accepted
            self.success = True
            self.message = "coupled mode-extension probe reached the endpoint"
        else:
            self.fixed_potential, self.fixed_inertia, self.fixed_stiffness = (
                values.copy() for values in saved
            )
            self.success = False
            self.message = self.mode_extension_diagnostics[
                "termination_reason"
            ]
        return self

    def continue_spectral_fixed_point(
        self,
        initial,
        strengths,
        *,
        inertia=None,
        stiffness=None,
        modes=5,
        tolerance=1.0e-7,
        max_evaluations=600,
    ):
        """Track a coupled spectral fixed point through Feshbach strength."""
        potential = self.grid._field_values(initial, "initial")
        temporal = (
            np.ones_like(self.field)
            if inertia is None
            else self.grid._field_values(inertia, "inertia")
        )
        spatial = (
            np.ones_like(self.field)
            if stiffness is None
            else self.grid._field_values(stiffness, "stiffness")
        )
        history = []
        for strength in np.asarray(strengths, dtype=float):
            self.kernel.set_feshbach_strength(strength)
            self.solve_spectral_fixed_point(
                potential,
                inertia=temporal,
                stiffness=spatial,
                modes=modes,
                tolerance=tolerance,
                max_evaluations=max_evaluations,
            )
            history.append(
                {
                    "strength": float(strength),
                    "success": bool(self.success),
                    "message": self.message,
                    "max_residual": float(np.max(np.abs(self.fixed_beta))),
                    "potential": self.fixed_potential.copy(),
                    "inertia": self.fixed_inertia.copy(),
                    "stiffness": self.fixed_stiffness.copy(),
                    "eta_t": float(self.geometry["eta_t"]),
                    "eta_x": float(self.geometry["eta_x"]),
                    "dynamic_exponent": float(self.geometry["dynamic_exponent"]),
                }
            )
            if not self.success:
                break
            potential = self.fixed_potential
            temporal = self.fixed_inertia
            spatial = self.fixed_stiffness
        self.continuation = history
        return self

    def stability_spectrum(
        self,
        *,
        modes=5,
        step=2.0e-5,
        project_redundant=True,
        redundancy_tolerance=1.0e-10,
        redundancy_invariance_tolerance=5.0e-2,
    ):
        """Linearize the normalized flow and audit field reparameterization.

        A direction is removed only when it is approximately invariant under
        the truncated stability matrix.  A large overlap with one eigenvector
        is not sufficient: projecting a non-invariant direction changes the
        spectrum instead of taking a legitimate redundant quotient.
        """
        if self.fixed_potential is None:
            raise ValueError("solve a fixed point before computing its spectrum.")
        if self.fixed_inertia is None or self.fixed_stiffness is None:
            raise ValueError("the stability spectrum requires a coupled fixed point.")
        modes = int(modes)
        step = float(step)
        if modes < 2 or 3 * modes > self.field.size:
            raise ValueError("modes must satisfy 2 <= 3*modes <= field.size.")
        if step <= 0.0:
            raise ValueError("step must be positive.")

        coordinate = self.field / np.max(np.abs(self.field))
        center = self.normalize_index
        columns = []
        for degree in 2 * np.arange(1, modes):
            coefficients = np.zeros(degree + 1)
            coefficients[degree] = 1.0
            values = np.polynomial.chebyshev.chebval(coordinate, coefficients)
            columns.append(values - values[center])
        basis = np.column_stack(columns)
        projector = np.linalg.pinv(basis)
        block = modes - 1
        size = 3 * block

        def vector_field(parameters):
            potential = self.fixed_potential + basis @ parameters[:block]
            inertia = self.fixed_inertia * np.exp(
                basis @ parameters[block : 2 * block]
            )
            stiffness = self.fixed_stiffness * np.exp(
                basis @ parameters[2 * block :]
            )
            potential_rate, inertia_rate, stiffness_rate = self.rates(
                potential, inertia=inertia, stiffness=stiffness
            )
            potential_rate = potential_rate - potential_rate[center]
            return np.concatenate(
                (
                    projector @ potential_rate,
                    projector @ (inertia_rate / inertia),
                    projector @ (stiffness_rate / stiffness),
                )
            )

        jacobian = np.empty((size, size))
        origin = np.zeros(size)
        for index in range(size):
            displacement = np.zeros(size)
            displacement[index] = step
            jacobian[:, index] = (
                vector_field(origin + displacement)
                - vector_field(origin - displacement)
            ) / (2.0 * step)
        full_eigenvalues, full_eigenvectors = np.linalg.eig(jacobian)
        full_order = np.argsort(full_eigenvalues.real)[::-1]
        full_eigenvalues = full_eigenvalues[full_order]
        full_eigenvectors = full_eigenvectors[:, full_order]

        field_rescaling = np.concatenate(
            (
                projector
                @ (self.field * self.derivative(self.fixed_potential, 1)),
                projector
                @ (
                    self.field
                    * self.derivative(np.log(self.fixed_inertia), 1)
                ),
                projector
                @ (
                    self.field
                    * self.derivative(np.log(self.fixed_stiffness), 1)
                ),
            )
        )
        redundant_norm = float(np.linalg.norm(field_rescaling))
        rayleigh = float("nan")
        invariance_residual = float("nan")
        relative_invariance_residual = float("inf")
        if redundant_norm > redundancy_tolerance:
            direction = field_rescaling / redundant_norm
            image = jacobian @ direction
            rayleigh = float(np.dot(direction, image))
            invariance_residual = float(
                np.linalg.norm(image - rayleigh * direction)
            )
            relative_invariance_residual = invariance_residual / max(
                float(np.linalg.norm(image)), redundancy_tolerance
            )
        invariant = bool(
            redundant_norm > redundancy_tolerance
            and relative_invariance_residual
            <= float(redundancy_invariance_tolerance)
        )
        physical_basis = np.eye(size)
        redundant_rank = 0
        if project_redundant and invariant:
            redundant = field_rescaling[:, None] / redundant_norm
            left, singular_values, _ = np.linalg.svd(
                redundant, full_matrices=True
            )
            redundant_rank = int(
                np.sum(singular_values > redundancy_tolerance)
            )
            physical_basis = left[:, redundant_rank:]
        physical_matrix = physical_basis.T @ jacobian @ physical_basis
        eigenvalues, eigenvectors = np.linalg.eig(physical_matrix)
        order = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = physical_basis @ eigenvectors[:, order]

        overlaps = np.zeros(full_eigenvalues.size)
        if redundant_norm > redundancy_tolerance:
            direction = field_rescaling / redundant_norm
            overlaps = np.abs(
                np.conjugate(full_eigenvectors).T @ direction
            ) / np.linalg.norm(full_eigenvectors, axis=0)
        dominant = int(np.argmax(overlaps)) if overlaps.size else -1
        self.stability_matrix = jacobian
        self.stability_full_eigenvalues = full_eigenvalues
        self.stability_full_eigenvectors = full_eigenvectors
        self.stability_eigenvalues = eigenvalues
        self.stability_eigenvectors = eigenvectors
        self.stability_physical_basis = physical_basis
        self.stability_redundant_direction = field_rescaling
        self.stability_redundant_overlaps = overlaps
        mode_diagnostics = []
        for index, (value, vector) in enumerate(
            zip(eigenvalues, eigenvectors.T)
        ):
            weights = np.array(
                [
                    np.linalg.norm(vector[:block]) ** 2,
                    np.linalg.norm(vector[block : 2 * block]) ** 2,
                    np.linalg.norm(vector[2 * block :]) ** 2,
                ]
            )
            weights /= max(float(np.sum(weights)), redundancy_tolerance)
            overlap = (
                float(
                    abs(np.vdot(vector, field_rescaling / redundant_norm))
                    / np.linalg.norm(vector)
                )
                if redundant_norm > redundancy_tolerance
                else 0.0
            )
            labels = ("potential", "temporal metric", "spatial metric")
            mode_diagnostics.append(
                {
                    "index": index,
                    "eigenvalue": value,
                    "relevant": bool(value.real > 1.0e-7),
                    "dominant_block": labels[int(np.argmax(weights))],
                    "potential_fraction": float(weights[0]),
                    "temporal_metric_fraction": float(weights[1]),
                    "spatial_metric_fraction": float(weights[2]),
                    "field_rescaling_overlap": overlap,
                    "redundant": bool(invariant and overlap >= 0.8),
                }
            )
        self.stability_mode_diagnostics = mode_diagnostics
        self.redundancy_diagnostics = {
            "vacuum_energy_gauge_fixed": True,
            "field_rescaling_projection_requested": bool(project_redundant),
            "field_rescaling_projected": bool(redundant_rank),
            "field_rescaling_is_invariant": invariant,
            "field_rescaling_norm": redundant_norm,
            "field_rescaling_rayleigh_exponent": rayleigh,
            "field_rescaling_invariance_residual": invariance_residual,
            "field_rescaling_relative_invariance_residual": (
                relative_invariance_residual
            ),
            "projection_warning": (
                None
                if not project_redundant or invariant
                else "field-rescaling candidate is not an invariant stability "
                "direction and was not projected"
            ),
            "dominant_full_mode": dominant,
            "dominant_full_mode_overlap": (
                float(overlaps[dominant]) if dominant >= 0 else 0.0
            ),
            "dominant_full_mode_eigenvalue": (
                full_eigenvalues[dominant] if dominant >= 0 else np.nan
            ),
        }
        relevant = self.stability_eigenvalues[
            self.stability_eigenvalues.real > 1.0e-7
        ]
        self.relevant_mode_count = int(relevant.size)
        self.relevant_eigenvalue = (
            float(relevant[0].real) if relevant.size else float("nan")
        )
        self.correlation_exponent = 1.0 / self.relevant_eigenvalue
        return self.stability_eigenvalues

    def fit_self_energy(self, potential, momenta, *, rank=None, tolerance=1e-6):
        """Sample and compress the static Feshbach self-energy Sigma(phi, q)."""
        potential = self.grid._field_values(potential, "potential")
        momenta = np.asarray(momenta, dtype=float)
        if momenta.ndim != 1 or momenta.size < 2:
            raise ValueError("momenta must be a one-dimensional grid.")
        curvature = self.derivative(potential, 2)
        cubic = self.derivative(potential, 3)
        quartic = self.derivative(potential, 4)
        values = np.empty((self.field.size, momenta.size))
        for field_index in range(self.field.size):
            frame = {
                "curvature": float(curvature[field_index]),
                "gap2": self.kernel.cutoff**2 + float(curvature[field_index]),
                "curvature_derivative": float(cubic[field_index]),
            }
            couplings = Phi4GaussianCouplings(
                quartic=float(quartic[field_index])
            )
            for momentum_index, momentum in enumerate(momenta):
                values[field_index, momentum_index] = (
                    self.kernel._static_response(
                        frame, couplings, self.kernel.cutoff, momentum
                    )
                )
        left, singular_values, right = np.linalg.svd(values, full_matrices=False)
        if rank is None:
            squared = singular_values**2
            discarded = np.sqrt(np.cumsum(squared[::-1])[::-1])
            scale = np.linalg.norm(singular_values)
            rank = singular_values.size
            for candidate in range(1, singular_values.size + 1):
                error = discarded[candidate] if candidate < discarded.size else 0.0
                if error <= tolerance * scale:
                    rank = candidate
                    break
        rank = int(rank)
        if not 1 <= rank <= singular_values.size:
            raise ValueError("rank lies outside the available SVD range.")
        fitted = (left[:, :rank] * singular_values[:rank]) @ right[:rank]
        self.self_energy = {
            "field": self.field.copy(),
            "momentum": momenta.copy(),
            "values": values,
            "fitted": fitted,
            "left": left[:, :rank],
            "singular_values": singular_values[:rank],
            "right": right[:rank],
            "rank": rank,
            "relative_error": float(
                np.linalg.norm(values - fitted) / np.linalg.norm(values)
            ),
        }
        return self.self_energy


class Phi4CovariantFRG:
    r"""Smooth covariant Wetterich FRG at quartic potential order.

    The flow uses ``ell = log(Lambda_0 / Lambda)`` and a regulator on the full
    Euclidean momentum.  The current external-momentum quadrature covers
    ``D=2``; radial potential thresholds are formulated for general ``D``.
    """

    def __init__(
        self,
        spacetime_dimension: int = 2,
        *,
        radial_order: int = 120,
        angular_order: int = 80,
        radial_max: float = 35.0,
        regulator=None,
    ):
        self.spacetime_dimension = int(spacetime_dimension)
        self.radial_order = int(radial_order)
        self.angular_order = int(angular_order)
        self.radial_max = float(radial_max)
        self.regulator = ExponentialRegulator() if regulator is None else regulator
        if self.spacetime_dimension < 2:
            raise ValueError("spacetime_dimension must be at least two.")
        if self.radial_order < 8 or self.angular_order < 8:
            raise ValueError("quadrature orders must be at least eight.")
        if self.radial_max <= 0.0:
            raise ValueError("radial_max must be positive.")

        nodes, weights = leggauss(self.radial_order)
        self._momentum2 = 0.5 * self.radial_max * (nodes + 1.0)
        self._radial_weights = 0.5 * self.radial_max * weights
        self._regulator = self.regulator.value(self._momentum2)
        angular_power = 0.5 * (self.spacetime_dimension - 3.0)
        cosines, angular_weights = roots_jacobi(
            self.angular_order, angular_power, angular_power
        )
        self._angular_cosines = cosines
        self._angular_weights = angular_weights / np.sum(angular_weights)

    @property
    def radial_measure(self) -> float:
        dimension = self.spacetime_dimension
        return 0.5 * _sphere_area(dimension) / (2.0 * pi) ** dimension

    def _thresholds(self, curvature: float, eta: float, powers):
        curvature = float(curvature)
        denominator = self._momentum2 + self._regulator + curvature
        if np.any(denominator <= 0.0):
            raise ValueError("the regulated Gaussian propagator is unstable.")
        dimension = self.spacetime_dimension
        radial_power = self._momentum2 ** (0.5 * dimension - 1.0)
        shell_weight = self.regulator.scale_derivative(
            self._momentum2, eta
        )
        weights = self._radial_weights * radial_power * shell_weight
        return np.array(
            [
                self.radial_measure * np.sum(weights / denominator**power)
                for power in powers
            ]
        )

    def thresholds(self, curvature, *, eta=0.0, powers=(1,)):
        """Return standard regulator threshold functions for local curvature."""
        return self._thresholds(curvature, eta, tuple(powers))

    def two_point_kernel(
        self,
        field,
        couplings: Phi4GaussianCouplings,
        external_momentum,
        *,
        eta=0.0,
    ) -> float:
        r"""Return the regulated ``G(P)^2 G(P+q)`` shell kernel."""
        external_momentum = np.asarray(external_momentum, dtype=float)
        expected = (self.spacetime_dimension,)
        if external_momentum.shape != expected:
            raise ValueError(f"external_momentum must have shape {expected}.")
        curvature = float(Phi4GaussianShell.curvature(field, couplings))
        return self._two_point_kernel(
            curvature, np.linalg.norm(external_momentum), eta=eta
        )

    def _two_point_kernel(self, curvature, external_momentum, *, eta=0.0):
        """Return the isotropic shell kernel for local two-point data."""
        curvature = float(curvature)
        external_momentum = float(external_momentum)
        denominator = self._momentum2 + self._regulator + curvature
        if np.any(denominator <= 0.0):
            raise ValueError("the regulated Gaussian propagator is unstable.")
        momentum = np.sqrt(self._momentum2)
        shifted2 = (
            self._momentum2[:, None]
            + external_momentum**2
            + 2.0
            * momentum[:, None]
            * external_momentum
            * self._angular_cosines[None, :]
        )
        shifted_denominator = (
            shifted2 + self.regulator.value(shifted2) + curvature
        )
        if np.any(shifted_denominator <= 0.0):
            raise ValueError("the shifted regulated propagator is unstable.")
        shifted_propagator = (
            1.0 / shifted_denominator
        ) @ self._angular_weights
        shell_weight = self.regulator.scale_derivative(
            self._momentum2, eta
        )
        radial_power = self._momentum2 ** (
            0.5 * self.spacetime_dimension - 1.0
        )
        integrand = (
            self._radial_weights
            * radial_power
            * shell_weight
            * shifted_propagator
            / denominator**2
        )
        return float(self.radial_measure * np.sum(integrand))

    def kinetic_rate(
        self,
        field,
        couplings: Phi4GaussianCouplings,
        *,
        eta=0.0,
        axis: int = 0,
        momentum_steps=None,
    ) -> float:
        r"""Project ``partial_ell Gamma^(2)(q)`` onto ``q^2``."""
        if not 0 <= axis < self.spacetime_dimension:
            raise ValueError(
                "axis must index one of the Euclidean momentum dimensions."
            )
        if momentum_steps is None:
            momentum_steps = np.array([0.006, 0.009, 0.013, 0.018, 0.025])
        momentum_steps = np.atleast_1d(np.asarray(momentum_steps, dtype=float))
        if np.any(momentum_steps <= 0.0):
            raise ValueError("momentum_steps must be positive.")
        origin = self.two_point_kernel(
            field,
            couplings,
            np.zeros(self.spacetime_dimension),
            eta=eta,
        )
        slopes = np.empty(momentum_steps.size)
        for index, step in enumerate(momentum_steps):
            external = np.zeros(self.spacetime_dimension)
            external[axis] = step
            value = self.two_point_kernel(
                field, couplings, external, eta=eta
            )
            slopes[index] = (value - origin) / step**2
        if momentum_steps.size == 1:
            slope = slopes[0]
        else:
            order = min(2, momentum_steps.size - 1)
            slope = np.polynomial.polynomial.polyfit(
                momentum_steps**2, slopes, order
            )[0]
        vertex2 = float(Phi4GaussianShell.third_derivative(field, couplings)) ** 2
        return float(-vertex2 * slope)

    def local_kinetic_rate(
        self,
        curvature,
        cubic,
        *,
        eta=0.0,
        wavefunction=1.0,
    ) -> float:
        r"""Project the local two-point flow analytically onto ``q^2``.

        ``wavefunction`` is the local value of ``Z(rho)`` in the inverse
        propagator.  Field derivatives of ``Z`` are deliberately omitted;
        they are separate momentum-dependent vertices in the local DE2
        closure.
        """
        curvature = float(curvature)
        cubic = float(cubic)
        wavefunction = float(wavefunction)
        if wavefunction <= 0.0:
            raise ValueError("wavefunction must be positive.")
        momentum2 = self._momentum2
        denominator = (
            wavefunction * momentum2 + self._regulator + curvature
        )
        if np.any(denominator <= 0.0):
            raise ValueError("the regulated Gaussian propagator is unstable.")
        first = wavefunction + self.regulator.first_derivative(momentum2)
        second = self.regulator.second_derivative(momentum2)
        shifted_first = -first / denominator**2
        shifted_second = (
            2.0 * first**2 / denominator**3 - second / denominator**2
        )
        momentum_coefficient = shifted_first + (
            2.0 * momentum2 / self.spacetime_dimension
        ) * shifted_second
        radial_power = momentum2 ** (0.5 * self.spacetime_dimension - 1.0)
        shell_weight = self.regulator.scale_derivative(momentum2, eta)
        integral = self.radial_measure * np.sum(
            self._radial_weights
            * radial_power
            * shell_weight
            * momentum_coefficient
            / denominator**2
        )
        return float(-cubic**2 * integral)

    def local_anomalous_dimension(self, curvature, cubic) -> float:
        """Return the self-consistent local anomalous dimension."""
        if cubic == 0.0:
            return 0.0
        rate0 = self.local_kinetic_rate(curvature, cubic, eta=0.0)
        rate1 = self.local_kinetic_rate(curvature, cubic, eta=1.0)
        eta_slope = rate1 - rate0
        if np.isclose(eta_slope, 1.0):
            raise ValueError("the anomalous-dimension closure is singular.")
        return float(rate0 / (1.0 - eta_slope))

    def anomalous_dimension(self, couplings: Phi4GaussianCouplings) -> float:
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("anomalous_dimension requires a Z2-symmetric potential.")
        if couplings.quartic < 0.0:
            raise ValueError("quartic must be nonnegative.")
        if couplings.mass2 >= 0.0 or couplings.quartic == 0.0:
            return 0.0
        minimum = np.sqrt(
            -6.0 * couplings.mass2 / couplings.quartic
        )
        return self.local_anomalous_dimension(
            Phi4GaussianShell.curvature(minimum, couplings),
            Phi4GaussianShell.third_derivative(minimum, couplings),
        )

    def beta(self, couplings: Phi4GaussianCouplings):
        r"""Return quartic beta functions and the self-consistent ``eta``."""
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("beta requires a Z2-symmetric potential.")
        if couplings.quartic < 0.0:
            raise ValueError("quartic must be nonnegative.")
        eta = self.anomalous_dimension(couplings)
        _, threshold2, threshold3 = self._thresholds(
            couplings.mass2, eta, (1, 2, 3)
        )
        dimension = self.spacetime_dimension
        mass2 = (2.0 - eta) * couplings.mass2
        mass2 += 0.5 * couplings.quartic * threshold2
        quartic = (4.0 - dimension - 2.0 * eta) * couplings.quartic
        quartic -= 3.0 * couplings.quartic**2 * threshold3
        return Phi4GaussianCouplings(mass2=mass2, quartic=quartic), eta

    def beta_potential(self, field, couplings: Phi4GaussianCouplings):
        r"""Evaluate the covariant regulated flow of the local potential."""
        field = np.asarray(field, dtype=float)
        eta = self.anomalous_dimension(couplings)
        field_dimension = 0.5 * (
            self.spacetime_dimension - 2.0 + eta
        )
        derivative = (
            couplings.source
            + couplings.mass2 * field
            + 0.5 * couplings.cubic * field**2
            + couplings.quartic * field**3 / 6.0
        )
        loop = np.empty_like(field)
        for index in np.ndindex(field.shape):
            loop[index] = -0.5 * self._thresholds(
                Phi4GaussianShell.curvature(field[index], couplings), eta, (1,)
            )[0]
        return (
            self.spacetime_dimension
            * Phi4GaussianShell.potential(field, couplings)
            - field_dimension * field * derivative
            + loop
        )


class Phi4FRG:
    r"""Standard functional RG in a running-minimum potential expansion.

    The dimensionless effective potential is represented as

    ``u(rho) = sum_{n=2}^N lambda_n (rho-kappa)^n / n!``

    with ``rho = phi^2 / 2``.  ``approximation='lpa'`` keeps ``eta=0`` and
    ``approximation='lpa_prime'`` obtains it from the external-momentum
    projection.  ``approximation='de2'`` additionally represents

    ``Z(rho) = 1 + sum_{n=1}^M z_n (rho-kappa)^n / n!``.

    The DE2 kinetic source retains the local potential-vertex bubble and the
    local ``Z``-dressed propagator.  Vertices generated by derivatives of
    ``Z`` are not included, so this is a local DE2 closure rather than the
    complete second-order derivative expansion.  The flow parameter is
    ``ell = log(Lambda_0 / Lambda)``.
    """

    def __init__(
        self,
        order: int = 6,
        approximation: str = "lpa_prime",
        *,
        spacetime_dimension: int = 3,
        radial_order: int = 100,
        angular_order: int = 64,
        radial_max: float = 35.0,
        regulator=None,
        wavefunction_order=None,
    ):
        self.order = int(order)
        if self.order < 2:
            raise ValueError("order must be at least two.")
        approximation = (
            str(approximation).lower().replace("'", "_prime").replace("-", "_")
        )
        if approximation == "local_de2":
            approximation = "de2"
        if approximation not in {"lpa", "lpa_prime", "de2"}:
            raise ValueError(
                "approximation must be 'lpa', 'lpa_prime', or 'de2'."
            )
        if wavefunction_order is None:
            wavefunction_order = 1 if approximation == "de2" else 0
        self.wavefunction_order = int(wavefunction_order)
        if self.wavefunction_order < 0:
            raise ValueError("wavefunction_order must be nonnegative.")
        if approximation == "de2" and self.wavefunction_order < 1:
            raise ValueError("de2 requires wavefunction_order >= 1.")
        if approximation != "de2" and self.wavefunction_order != 0:
            raise ValueError("wavefunction_order is only used by de2.")
        self.approximation = approximation
        self.kernel = Phi4CovariantFRG(
            spacetime_dimension=spacetime_dimension,
            radial_order=radial_order,
            angular_order=angular_order,
            radial_max=radial_max,
            regulator=regulator,
        )
        self.fixed_state = None
        self.fixed_beta = None
        self.fixed_eta = None
        self.fixed_point_history = None
        self.stability_matrix = None
        self.stability_eigenvalues = None
        self.relevant_eigenvalue = None
        self.correlation_exponent = None
        self.success = None
        self.message = None

    @property
    def spacetime_dimension(self) -> int:
        return self.kernel.spacetime_dimension

    @property
    def state_size(self) -> int:
        return self.order + self.wavefunction_order

    def _state(self, state, order=None, wavefunction_order=None):
        order = self.order if order is None else int(order)
        wavefunction_order = (
            self.wavefunction_order
            if wavefunction_order is None
            else int(wavefunction_order)
        )
        state = np.asarray(state, dtype=float)
        if state.shape != (order + wavefunction_order,):
            raise ValueError(
                f"state must contain kappa, lambda_2 through lambda_{order}, "
                f"and {wavefunction_order} wavefunction coefficients."
            )
        if not np.all(np.isfinite(state)):
            raise ValueError("state must be finite.")
        if state[0] < 0.0:
            raise ValueError("the running minimum kappa must be nonnegative.")
        if state[1] <= 0.0:
            raise ValueError("lambda_2 must be positive.")
        return state

    def potential(self, field, state=None, *, offset=0.0):
        """Evaluate the running-minimum polynomial potential on ``field``."""
        if state is None:
            if self.fixed_state is None:
                raise ValueError("state is required before a fixed point is solved.")
            state = self.fixed_state
        state = self._state(state)
        rho_displacement = 0.5 * np.asarray(field, dtype=float) ** 2 - state[0]
        values = np.full_like(rho_displacement, float(offset), dtype=float)
        for degree, vertex in enumerate(state[1 : self.order], start=2):
            values += vertex * rho_displacement**degree / factorial(degree)
        return values

    def wavefunction(self, field, state=None):
        r"""Evaluate ``Z(rho)`` with the normalization ``Z(kappa)=1``."""
        if state is None:
            if self.fixed_state is None:
                raise ValueError("state is required before a fixed point is solved.")
            state = self.fixed_state
        state = self._state(state)
        displacement = 0.5 * np.asarray(field, dtype=float) ** 2 - state[0]
        values = np.ones_like(displacement, dtype=float)
        for degree in range(1, self.wavefunction_order + 1):
            values += (
                state[self.order + degree - 1]
                * displacement**degree
                / factorial(degree)
            )
        return values

    def quartic_state(self, couplings: Phi4GaussianCouplings):
        """Convert a broken-phase quartic potential to this expansion."""
        if couplings.source != 0.0 or couplings.cubic != 0.0:
            raise ValueError("quartic_state requires a Z2-symmetric potential.")
        if couplings.mass2 >= 0.0 or couplings.quartic <= 0.0:
            raise ValueError("quartic_state requires mass2 < 0 and quartic > 0.")
        state = np.zeros(self.state_size)
        state[0] = -3.0 * couplings.mass2 / couplings.quartic
        state[1] = couplings.quartic / 3.0
        return state

    def _lpa_anomalous_dimension(self, state, order) -> float:
        state = self._state(state, order, 0)
        if self.approximation == "lpa":
            return 0.0
        kappa = state[0]
        lambda2 = state[1]
        lambda3 = state[2] if order >= 3 else 0.0
        if kappa == 0.0:
            return 0.0
        curvature = 2.0 * kappa * lambda2
        cubic = np.sqrt(2.0 * kappa) * (
            3.0 * lambda2 + 2.0 * kappa * lambda3
        )
        return self.kernel.local_anomalous_dimension(curvature, cubic)

    def anomalous_dimension(self, state, *, order=None) -> float:
        """Return ``eta`` at the running minimum."""
        order = self.order if order is None else int(order)
        if self.approximation == "de2":
            state = self._state(state, order, self.wavefunction_order)
            return self._de2_beta_for_order(
                state, order, self.wavefunction_order
            )[1]
        return self._lpa_anomalous_dimension(state, order)

    def _flow_polynomial(self, state, order):
        state = self._state(state, order, 0)
        kappa = state[0]
        vertices = np.zeros(order + 2)
        vertices[2 : order + 1] = state[1:]
        coefficients = np.zeros(order + 2)
        coefficients[2 : order + 1] = vertices[2 : order + 1] / np.array(
            [factorial(degree) for degree in range(2, order + 1)]
        )
        potential = np.polynomial.Polynomial(coefficients)
        rho = np.polynomial.Polynomial([kappa, 1.0])
        curvature = potential.deriv(1) + 2.0 * rho * potential.deriv(2)
        local_curvature = float(curvature.coef[0])
        eta = self._lpa_anomalous_dimension(state, order)
        thresholds = self.kernel.thresholds(
            local_curvature,
            eta=eta,
            powers=range(1, order + 2),
        )
        displacement = curvature - np.polynomial.Polynomial([local_curvature])
        loop = np.polynomial.Polynomial([0.0])
        power = np.polynomial.Polynomial([1.0])
        for degree in range(order + 1):
            loop -= 0.5 * (-1.0) ** degree * thresholds[degree] * power
            power = (power * displacement).truncate(order + 2)
        scaling_dimension = self.spacetime_dimension - 2.0 + eta
        flow = (
            self.spacetime_dimension * potential
            - scaling_dimension * rho * potential.deriv(1)
            + loop
        ).truncate(order + 2)
        return flow, vertices, eta

    def _beta_for_order(self, state, order):
        flow, vertices, eta = self._flow_polynomial(state, order)
        coefficients = np.pad(
            flow.coef,
            (0, max(0, order + 2 - flow.coef.size)),
        )
        kappa_rate = -coefficients[1] / vertices[2]
        rates = np.empty(order)
        rates[0] = kappa_rate
        for degree in range(2, order + 1):
            rates[degree - 1] = (
                factorial(degree) * coefficients[degree]
                + vertices[degree + 1] * kappa_rate
            )
        return rates, eta

    @staticmethod
    def _batch_polynomial_product(left, right):
        """Multiply batches of coefficient-order polynomials and truncate."""
        size = left.shape[1]
        product = np.zeros_like(left)
        for degree in range(size):
            product[:, degree:] += (
                left[:, : size - degree] * right[:, degree, None]
            )
        return product

    def _de2_potential_flow(self, state, eta, order, wavefunction_order):
        state = self._state(state, order, wavefunction_order)
        kappa = state[0]
        vertices = np.zeros(order + 2)
        vertices[2 : order + 1] = state[1:order]
        potential_coefficients = np.zeros(order + 1)
        for degree in range(2, order + 1):
            potential_coefficients[degree] = (
                vertices[degree] / factorial(degree)
            )
        potential = np.polynomial.Polynomial(potential_coefficients)
        wavefunction_coefficients = np.zeros(wavefunction_order + 1)
        wavefunction_coefficients[0] = 1.0
        for degree in range(1, wavefunction_order + 1):
            wavefunction_coefficients[degree] = (
                state[order + degree - 1] / factorial(degree)
            )
        wavefunction = np.polynomial.Polynomial(wavefunction_coefficients)
        rho = np.polynomial.Polynomial([kappa, 1.0])
        curvature = potential.deriv(1) + 2.0 * rho * potential.deriv(2)

        size = order + 1
        curvature_coefficients = np.pad(
            curvature.coef, (0, max(0, size - curvature.coef.size))
        )[:size]
        local_curvature = curvature_coefficients[0]
        curvature_coefficients[0] = 0.0
        delta_wavefunction = np.zeros(size)
        count = min(size, wavefunction.coef.size)
        delta_wavefunction[:count] = wavefunction.coef[:count]
        delta_wavefunction[0] -= 1.0

        momentum2 = self.kernel._momentum2
        denominator = momentum2 + self.kernel._regulator + local_curvature
        if np.any(denominator <= 0.0):
            raise ValueError("the regulated Gaussian propagator is unstable.")
        displacement = (
            curvature_coefficients[None, :]
            + momentum2[:, None] * delta_wavefunction[None, :]
        )
        radial_power = momentum2 ** (0.5 * self.spacetime_dimension - 1.0)
        shell_weight = self.kernel.regulator.scale_derivative(momentum2, eta)
        weights = self.kernel._radial_weights * radial_power * shell_weight
        loop_coefficients = np.zeros(size)
        power = np.zeros_like(displacement)
        power[:, 0] = 1.0
        for degree in range(order + 1):
            factor = (
                -0.5
                * self.kernel.radial_measure
                * (-1.0) ** degree
                * weights
                / denominator ** (degree + 1)
            )
            loop_coefficients += np.sum(factor[:, None] * power, axis=0)
            power = self._batch_polynomial_product(power, displacement)

        scaling_dimension = self.spacetime_dimension - 2.0 + eta
        flow = (
            self.spacetime_dimension * potential
            - scaling_dimension * rho * potential.deriv(1)
            + np.polynomial.Polynomial(loop_coefficients)
        ).truncate(size)
        return flow, vertices, potential, wavefunction

    def _de2_kinetic_source(
        self, potential, wavefunction, kappa, eta, wavefunction_order
    ):
        span = min(0.45 * max(kappa, 1.0e-4), 0.03)
        lower = -min(span, 0.9 * kappa) if kappa > 0.0 else 0.0
        sample_count = max(7, 2 * wavefunction_order + 5)
        displacement = np.linspace(lower, span, sample_count)
        rho = kappa + displacement
        first = potential.deriv(1)(displacement)
        second = potential.deriv(2)(displacement)
        third = potential.deriv(3)(displacement)
        curvature = first + 2.0 * rho * second
        cubic = np.sqrt(2.0 * rho) * (3.0 * second + 2.0 * rho * third)
        local_wavefunction = wavefunction(displacement)
        source = np.array(
            [
                self.kernel.local_kinetic_rate(
                    local_curvature,
                    local_cubic,
                    eta=eta,
                    wavefunction=local_z,
                )
                for local_curvature, local_cubic, local_z in zip(
                    curvature, cubic, local_wavefunction
                )
            ]
        )
        fit_order = min(wavefunction_order + 3, sample_count - 1)
        fitted = np.polynomial.Polynomial.fit(
            displacement, source, fit_order
        ).convert()
        coefficients = np.pad(
            fitted.coef,
            (0, max(0, wavefunction_order + 1 - fitted.coef.size)),
        )[: wavefunction_order + 1]
        coefficients[0] = self.kernel.local_kinetic_rate(
            potential.deriv(1)(0.0) + 2.0 * kappa * potential.deriv(2)(0.0),
            np.sqrt(2.0 * kappa)
            * (
                3.0 * potential.deriv(2)(0.0)
                + 2.0 * kappa * potential.deriv(3)(0.0)
            ),
            eta=eta,
            wavefunction=1.0,
        )
        return np.polynomial.Polynomial(coefficients)

    def _de2_raw_flow(self, state, eta, order, wavefunction_order):
        flow, vertices, potential, wavefunction = self._de2_potential_flow(
            state, eta, order, wavefunction_order
        )
        coefficients = np.pad(
            flow.coef, (0, max(0, order + 1 - flow.coef.size))
        )
        kappa_rate = -coefficients[1] / vertices[2]
        rates = np.empty(order + wavefunction_order)
        rates[0] = kappa_rate
        for degree in range(2, order + 1):
            rates[degree - 1] = (
                factorial(degree) * coefficients[degree]
                + vertices[degree + 1] * kappa_rate
            )

        kappa = state[0]
        source = self._de2_kinetic_source(
            potential, wavefunction, kappa, eta, wavefunction_order
        )
        rho = np.polynomial.Polynomial([kappa, 1.0])
        scaling_dimension = self.spacetime_dimension - 2.0 + eta
        wavefunction_flow = (
            source
            - eta * wavefunction
            - scaling_dimension * rho * wavefunction.deriv(1)
        ).truncate(wavefunction_order + 1)
        wavefunction_coefficients = np.pad(
            wavefunction_flow.coef,
            (
                0,
                max(
                    0,
                    wavefunction_order + 1 - wavefunction_flow.coef.size,
                ),
            ),
        )
        for degree in range(1, wavefunction_order + 1):
            next_vertex = (
                state[order + degree]
                if degree < wavefunction_order
                else 0.0
            )
            rates[order + degree - 1] = (
                factorial(degree) * wavefunction_coefficients[degree]
                + next_vertex * kappa_rate
            )
        normalization = wavefunction_coefficients[0]
        normalization += state[order] * kappa_rate
        return rates, float(normalization)

    def _de2_beta_for_order(self, state, order, wavefunction_order):
        state = self._state(state, order, wavefunction_order)
        rates0, normalization0 = self._de2_raw_flow(
            state, 0.0, order, wavefunction_order
        )
        rates1, normalization1 = self._de2_raw_flow(
            state, 1.0, order, wavefunction_order
        )
        slope = normalization1 - normalization0
        if np.isclose(slope, 0.0, atol=1.0e-12):
            raise ValueError("the DE2 normalization closure is singular.")
        eta = -normalization0 / slope
        rates = rates0 + eta * (rates1 - rates0)
        return rates, float(eta)

    def beta(self, state):
        r"""Return the coupling beta functions and ``eta``."""
        if self.approximation == "de2":
            return self._de2_beta_for_order(
                state, self.order, self.wavefunction_order
            )
        return self._beta_for_order(state, self.order)

    def _beta_at_order(self, state, order):
        return self._beta_for_order(state, int(order))

    def _initial_state(self):
        if self.spacetime_dimension == 2:
            return np.array([0.023, 125.0])
        if self.spacetime_dimension == 3:
            return np.array([0.03, 18.0])
        return np.array([0.03, 10.0])

    def solve_fixed_point(
        self,
        initial=None,
        *,
        tolerance=1.0e-8,
        max_evaluations=5000,
    ):
        """Continue the interacting fixed point through the configured order."""
        from scipy.optimize import least_squares, root

        if initial is None:
            supplied = self._initial_state()
        else:
            supplied = np.asarray(initial, dtype=float)
            allowed_sizes = {2, self.order, self.state_size}
            if supplied.ndim != 1 or supplied.size not in allowed_sizes:
                raise ValueError(
                    "initial must contain two, order, or state_size entries."
                )
        state = supplied[:2].copy()
        history = []
        self.success = True

        def optimize(values, evaluator, size):
            def residual(candidate):
                try:
                    rates, _ = evaluator(candidate)
                    if np.all(np.isfinite(rates)):
                        return rates
                except (ValueError, FloatingPointError, OverflowError):
                    pass
                return np.full(size, 1.0e6)

            solution = root(
                residual,
                values,
                method="hybr",
                options={
                    "xtol": min(tolerance, 1.0e-10),
                    "maxfev": max_evaluations,
                },
            )
            error = float(np.max(np.abs(residual(solution.x))))
            if error > 20.0 * tolerance:
                fallback = least_squares(
                    residual,
                    values,
                    x_scale=np.maximum(np.abs(values), 1.0),
                    xtol=tolerance,
                    ftol=tolerance,
                    gtol=tolerance,
                    max_nfev=max_evaluations,
                )
                fallback_error = float(np.max(np.abs(residual(fallback.x))))
                if fallback_error < error:
                    solution = fallback
                    error = fallback_error
            try:
                rates, eta = evaluator(solution.x)
            except ValueError:
                rates = residual(solution.x)
                eta = float("nan")
            return solution.x, rates, eta, error

        for order in range(2, self.order + 1):
            if order > 2:
                hint = supplied[order - 1] if supplied.size >= self.order else 0.0
                state = np.append(state, hint)
            evaluator = lambda values, stage=order: self._beta_at_order(
                values, stage
            )
            state, rates, eta, error = optimize(state, evaluator, order)
            converged = bool(error <= 20.0 * tolerance)
            history.append(
                {
                    "order": order,
                    "wavefunction_order": 0,
                    "state": state.copy(),
                    "eta": float(eta),
                    "max_residual": error,
                    "success": converged,
                }
            )
            if not converged:
                self.success = False
                self.message = (
                    f"fixed-point continuation failed at order {order}; "
                    f"residual={error:.3e}"
                )
                break

        if self.success and self.approximation == "de2":
            for wavefunction_order in range(1, self.wavefunction_order + 1):
                hint_index = self.order + wavefunction_order - 1
                hint = supplied[hint_index] if supplied.size == self.state_size else 0.0
                state = np.append(state, hint)
                evaluator = (
                    lambda values, stage=wavefunction_order:
                    self._de2_beta_for_order(values, self.order, stage)
                )
                size = self.order + wavefunction_order
                state, rates, eta, error = optimize(state, evaluator, size)
                converged = bool(error <= 20.0 * tolerance)
                history.append(
                    {
                        "order": self.order,
                        "wavefunction_order": wavefunction_order,
                        "state": state.copy(),
                        "eta": float(eta),
                        "max_residual": error,
                        "success": converged,
                    }
                )
                if not converged:
                    self.success = False
                    self.message = (
                        "fixed-point continuation failed at wavefunction order "
                        f"{wavefunction_order}; residual={error:.3e}"
                    )
                    break
        self.fixed_point_history = history
        self.fixed_state = state.copy()
        self.fixed_beta = rates.copy()
        self.fixed_eta = float(eta)
        if self.success:
            self.message = "fixed-point continuation converged."
            self.stability_spectrum()
        return self

    def stability_spectrum(self, *, step=2.0e-5):
        """Linearize the standard FRG flow and extract the relevant exponent."""
        if self.fixed_state is None:
            raise ValueError("solve a fixed point before computing stability.")
        step = float(step)
        if step <= 0.0:
            raise ValueError("step must be positive.")
        state = self.fixed_state
        scales = np.maximum(np.abs(state), 1.0)
        size = self.state_size
        jacobian = np.empty((size, size))
        for index in range(size):
            displacement = np.zeros(size)
            displacement[index] = step * scales[index]
            upper = self.beta(state + displacement)[0]
            lower = self.beta(state - displacement)[0]
            jacobian[:, index] = upper - lower
            jacobian[:, index] /= 2.0 * displacement[index]
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        ordering = np.argsort(eigenvalues.real)[::-1]
        self.stability_matrix = jacobian
        self.stability_eigenvalues = eigenvalues[ordering]
        self.stability_eigenvectors = eigenvectors[:, ordering]
        relevant = self.stability_eigenvalues[
            self.stability_eigenvalues.real > 1.0e-7
        ]
        self.relevant_eigenvalue = (
            float(relevant[0].real) if relevant.size else float("nan")
        )
        self.correlation_exponent = 1.0 / self.relevant_eigenvalue
        return self.stability_eigenvalues

    def run(self, initial, ell, *, method="BDF", rtol=1.0e-8, atol=1.0e-10):
        """Integrate a standard dimensionless FRG trajectory toward the IR."""
        from scipy.integrate import solve_ivp

        initial = self._state(initial)
        ell = np.asarray(ell, dtype=float)
        if ell.ndim != 1 or ell.size < 2 or np.any(np.diff(ell) <= 0.0):
            raise ValueError("ell must be a strictly increasing one-dimensional grid.")

        def equation(_ell, state):
            return self.beta(state)[0]

        solution = solve_ivp(
            equation,
            (float(ell[0]), float(ell[-1])),
            initial,
            method=method,
            t_eval=ell,
            rtol=rtol,
            atol=atol,
        )
        self.ell = solution.t
        self.history = solution.y.T
        self.state = solution.y[:, -1]
        self.eta_history = np.array(
            [self.anomalous_dimension(state) for state in self.history]
        )
        self.success = bool(solution.success)
        self.message = str(solution.message)
        return self


def gaussian_quantum_metric(mean_jacobian, covariance, covariance_derivatives):
    """Return the parameter-space metric of a real displaced Gaussian."""
    covariance = np.asarray(covariance, dtype=float)
    mean_jacobian = np.asarray(mean_jacobian, dtype=float)
    derivatives = np.asarray(covariance_derivatives, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if mean_jacobian.ndim != 2 or mean_jacobian.shape[1] != covariance.shape[0]:
        raise ValueError("mean_jacobian must have shape (nparams, nmodes).")
    expected = (mean_jacobian.shape[0],) + covariance.shape
    if derivatives.shape != expected:
        raise ValueError(f"covariance_derivatives must have shape {expected}.")
    inverse = np.linalg.inv(covariance)
    nparams = mean_jacobian.shape[0]
    metric = np.empty((nparams, nparams), dtype=float)
    for a in range(nparams):
        for b in range(nparams):
            displacement = 0.25 * mean_jacobian[a] @ inverse @ mean_jacobian[b]
            width = 0.125 * np.trace(
                inverse @ derivatives[a] @ inverse @ derivatives[b]
            )
            metric[a, b] = displacement + width
    return 0.5 * (metric + metric.T)


__all__ = [
    "ExponentialRegulator",
    "GaussianRegulator",
    "Phi4CovariantFRG",
    "Phi4FRG",
    "Phi4ContinuousQGRF",
    "Phi4FunctionalRegulatedQGRF",
    "Phi4FunctionalQGRG",
    "Phi4GaussianCouplings",
    "Phi4GaussianShell",
    "Phi4RegulatedQGRF",
    "Phi4SmoothQGRF",
    "Phi4VariationalQGRG",
    "Phi4WegnerHoughtonLPA",
    "gaussian_quantum_metric",
]
