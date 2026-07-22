"""Analytical response-form NARG maps.

The routines here replace a numerical diagonalize/truncate NARG step by a
closed response recursion.  The first target is a harmonic impurity coupled to
a Wilson chain, where the block is represented by its boundary susceptibility.

For multistate analytical NARG, the running object is a matrix PES
``K_l(phi)``.  Full nonadiabatic effects are represented by solving the active
``phi`` Hamiltonian in the moving conditional-state basis and dressing the
slow kinetic matrix by state overlaps,

``H_kin[(i,a),(j,b)] = T_phi[i,j] <A_i^a | A_j^b>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np


@dataclass
class AnalyticalHarmonicNARGStep:
    """One analytical Gaussian NARG response step."""

    site: int
    onsite: float
    coupling: float
    input_susceptibility: float
    curvature: float
    susceptibility: float
    block_scale: float = 1.0
    shell_scale: float = 1.0
    stable: bool = True


@dataclass
class AnalyticalHarmonicNARGResult:
    """Analytical NARG flow for a harmonic impurity plus Wilson chain."""

    impurity_stiffness: float
    initial_coupling: float
    steps: list[AnalyticalHarmonicNARGStep]
    nrg_rescaled: bool = False
    Lambda: float | None = None
    rescale_power: float = 1.0

    @property
    def curvatures(self):
        return np.asarray([step.curvature for step in self.steps], dtype=float)

    @property
    def susceptibilities(self):
        return np.asarray([step.susceptibility for step in self.steps], dtype=float)

    @property
    def stable(self):
        return all(step.stable for step in self.steps)


@dataclass
class AnalyticalHarmonicExponentEstimate:
    """Finite-Wilson-length exponent estimate from an analytical NARG flow."""

    stiffness: float
    perturbed_stiffness: float
    local_y: np.ndarray
    mean_y: float
    std_y: float
    window: np.ndarray


@dataclass
class AnalyticalLandauNARGStep:
    """One analytical Landau-PES NARG response step."""

    site: int
    onsite: float
    coupling: float
    input_stiffness: float
    input_quartic: float
    curvature: float
    quartic: float
    susceptibility: float
    quartic_response: float
    field_scale: float
    quartic_scale: float
    block_scale: float = 1.0
    shell_scale: float = 1.0
    stable: bool = True


@dataclass
class AnalyticalLandauNARGResult:
    """Analytical NARG flow for a Landau boundary generating function."""

    impurity_stiffness: float
    impurity_quartic: float
    initial_coupling: float
    steps: list[AnalyticalLandauNARGStep]
    nrg_rescaled: bool = False
    Lambda: float | None = None
    rescale_power: float = 1.0

    @property
    def curvatures(self):
        return np.asarray([step.curvature for step in self.steps], dtype=float)

    @property
    def quartics(self):
        return np.asarray([step.quartic for step in self.steps], dtype=float)

    @property
    def local_field_y(self):
        if self.Lambda is None:
            return np.full(len(self.steps), np.nan, dtype=float)
        scales = np.asarray([abs(step.field_scale) for step in self.steps], dtype=float)
        out = np.full(len(scales), np.nan, dtype=float)
        mask = scales > 1e-14
        out[mask] = np.log(scales[mask]) / np.log(float(self.Lambda))
        return out

    @property
    def local_quartic_y(self):
        if self.Lambda is None:
            return np.full(len(self.steps), np.nan, dtype=float)
        scales = np.asarray([abs(step.quartic_scale) for step in self.steps], dtype=float)
        out = np.full(len(scales), np.nan, dtype=float)
        mask = scales > 1e-14
        out[mask] = np.log(scales[mask]) / np.log(float(self.Lambda))
        return out

    @property
    def stable(self):
        return all(step.stable for step in self.steps)


@dataclass
class AnalyticalLandauCriticalExponents:
    """Closed Gaussian/Landau critical exponents for analytical NARG."""

    s: float
    y_t: float
    y_u: float
    y_h: float
    nu: float
    beta: float
    gamma: float
    delta: float
    hyperscaling_beta: float
    hyperscaling_gamma: float
    hyperscaling_delta: float


@dataclass
class ContinuumAnalyticalNARGState:
    """Continuum analytical-NARG scaling coordinates."""

    thermal: float
    quartic: float
    field: float = 0.0

    def asarray(self):
        return np.array([self.thermal, self.quartic, self.field], dtype=float)


@dataclass
class ContinuumAnalyticalNARGParameters:
    """Parameters for the continuum Landau aNARG beta functions."""

    s: float
    quartic_self: float = 1.0
    thermal_quartic: float = 0.0
    field_quartic: float = 0.0

    def __post_init__(self):
        self.s = float(self.s)
        self.quartic_self = float(self.quartic_self)
        self.thermal_quartic = float(self.thermal_quartic)
        self.field_quartic = float(self.field_quartic)
        if self.s <= 0.0:
            raise ValueError("s must be positive.")
        if self.quartic_self < 0.0:
            raise ValueError("quartic_self must be non-negative.")


@dataclass
class ContinuumAnalyticalNARGFlow:
    """Integrated continuum analytical-NARG trajectory."""

    parameters: ContinuumAnalyticalNARGParameters
    l: np.ndarray
    states: np.ndarray

    @property
    def thermal(self):
        return self.states[:, 0]

    @property
    def quartic(self):
        return self.states[:, 1]

    @property
    def field(self):
        return self.states[:, 2]


@dataclass
class ContinuumAnalyticalNARGLinearization:
    """Fixed-point linearization of the continuum analytical-NARG beta flow."""

    parameters: ContinuumAnalyticalNARGParameters
    fixed_point: ContinuumAnalyticalNARGState
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    exponents: AnalyticalLandauCriticalExponents


@dataclass
class CKLocalPotentialParameters:
    """Parameters for the CK local-potential RG flow."""

    s: float
    shell_measure: float = 1.0 / (2.0 * np.pi)

    def __post_init__(self):
        self.s = float(self.s)
        self.shell_measure = float(self.shell_measure)
        if self.s <= 0.0:
            raise ValueError("s must be positive.")
        if self.shell_measure < 0.0:
            raise ValueError("shell_measure must be non-negative.")


@dataclass
class CKLocalPotentialState:
    """Quartic truncation of the CK running potential."""

    mass: float
    quartic: float
    field: float = 0.0

    def asarray(self):
        return np.array([self.mass, self.quartic, self.field], dtype=float)


@dataclass
class CKLocalPotentialFlow:
    """Integrated CK local-potential RG trajectory."""

    parameters: CKLocalPotentialParameters
    l: np.ndarray
    states: np.ndarray

    @property
    def mass(self):
        return self.states[:, 0]

    @property
    def quartic(self):
        return self.states[:, 1]

    @property
    def field(self):
        return self.states[:, 2]


@dataclass
class CKLocalPotentialLinearization:
    """Fixed-point linearization of the CK local-potential RG."""

    parameters: CKLocalPotentialParameters
    fixed_point: CKLocalPotentialState
    jacobian: np.ndarray
    eigenvalues: np.ndarray


@dataclass
class OneModeNARGShellParameters:
    """Parameters for the one-mode eigenvalue NARG shell functional."""

    s: float
    shell_measure: float = 1.0 / (2.0 * np.pi)
    oscillator_frequency: float = 1.0
    fluctuation_amplitude: float = 2.0
    basis_size: int = 18
    fit_radius: float = 0.25
    n_fit_points: int = 17

    def __post_init__(self):
        self.s = float(self.s)
        self.shell_measure = float(self.shell_measure)
        self.oscillator_frequency = float(self.oscillator_frequency)
        self.fluctuation_amplitude = float(self.fluctuation_amplitude)
        self.basis_size = int(self.basis_size)
        self.fit_radius = float(self.fit_radius)
        self.n_fit_points = int(self.n_fit_points)
        if self.s <= 0.0:
            raise ValueError("s must be positive.")
        if self.shell_measure < 0.0:
            raise ValueError("shell_measure must be non-negative.")
        if self.oscillator_frequency <= 0.0:
            raise ValueError("oscillator_frequency must be positive.")
        if self.basis_size < 2:
            raise ValueError("basis_size must be at least two.")
        if self.fit_radius <= 0.0:
            raise ValueError("fit_radius must be positive.")
        if self.n_fit_points < 5:
            raise ValueError("n_fit_points must be at least five.")


@dataclass
class OneModeNARGShellProjection:
    """Quartic projection of one NARG eigenvalue shell."""

    parameters: OneModeNARGShellParameters
    state: CKLocalPotentialState
    grid: np.ndarray
    shell_energy: np.ndarray
    polynomial_coefficients: np.ndarray
    beta_shell: np.ndarray
    residual_norm: float


@dataclass
class OneModeNARGFlow:
    """Integrated one-mode eigenvalue NARG trajectory."""

    parameters: OneModeNARGShellParameters
    l: np.ndarray
    states: np.ndarray

    @property
    def mass(self):
        return self.states[:, 0]

    @property
    def quartic(self):
        return self.states[:, 1]

    @property
    def field(self):
        return self.states[:, 2]


@dataclass
class OneModeNARGLinearization:
    """Fixed-point linearization of the one-mode NARG beta flow."""

    parameters: OneModeNARGShellParameters
    fixed_point: CKLocalPotentialState
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    exponents: AnalyticalLandauCriticalExponents


@dataclass
class PolynomialPESState:
    """Taylor coefficients for a running local PES.

    The convention is ``U(phi) = sum_n c_n phi**n / n!`` with coefficients
    stored as ``[c_1, c_2, ...]``.  The spin-boson field convention is
    ``c_1 = -h``.
    """

    couplings: np.ndarray

    def __post_init__(self):
        self.couplings = np.asarray(self.couplings, dtype=float)
        if self.couplings.ndim != 1:
            raise ValueError("couplings must be one-dimensional.")
        if len(self.couplings) < 2:
            raise ValueError("at least c_1 and c_2 must be provided.")

    @property
    def order(self):
        return len(self.couplings)

    @property
    def field(self):
        return -float(self.couplings[0])

    @property
    def mass(self):
        return float(self.couplings[1])

    @property
    def quartic(self):
        return float(self.couplings[3]) if len(self.couplings) >= 4 else 0.0

    def asarray(self):
        return np.array(self.couplings, dtype=float, copy=True)

    @classmethod
    def from_ck(cls, state, *, order: int = 4):
        current = _coerce_ck_state(state)
        order = int(order)
        if order < 4:
            raise ValueError("order must be at least four.")
        couplings = np.zeros(order, dtype=float)
        couplings[0] = -current.field
        couplings[1] = current.mass
        couplings[3] = current.quartic
        return cls(couplings)


@dataclass
class PolynomialNARGShellProjection:
    """Higher-order Taylor projection of one NARG eigenvalue shell."""

    parameters: OneModeNARGShellParameters
    state: PolynomialPESState
    grid: np.ndarray
    shell_energy: np.ndarray
    shell_couplings: np.ndarray
    residual_norm: float


@dataclass
class PolynomialNARGFlow:
    """Integrated higher-order one-mode NARG trajectory."""

    parameters: OneModeNARGShellParameters
    l: np.ndarray
    states: np.ndarray

    @property
    def couplings(self):
        return self.states

    @property
    def field(self):
        return -self.states[:, 0]

    @property
    def mass(self):
        return self.states[:, 1]

    @property
    def quartic(self):
        if self.states.shape[1] < 4:
            return np.zeros(len(self.states), dtype=float)
        return self.states[:, 3]


@dataclass
class PolynomialNARGLinearization:
    """Fixed-point linearization of a higher-order one-mode NARG flow."""

    parameters: OneModeNARGShellParameters
    fixed_point: PolynomialPESState
    jacobian: np.ndarray
    eigenvalues: np.ndarray


@dataclass
class ConditionalOneModeNARGProjection:
    """Matrix-valued one-mode NARG shell in a retained conditional basis."""

    parameters: OneModeNARGShellParameters
    state: PolynomialPESState
    n_conditional_states: int
    reference_phi: float
    reference_energies: np.ndarray
    grid: np.ndarray
    shell_matrices: np.ndarray
    shell_couplings: np.ndarray
    residual_norm: float

    @property
    def surfaces(self):
        return np.linalg.eigvalsh(self.shell_matrices)

    @property
    def lowest_surface(self):
        return self.surfaces[:, 0]


@dataclass
class MatrixPolynomialPESState:
    """Matrix-valued Taylor PES for multi-state analytical NARG.

    The convention is ``K(phi) = sum_n C_n phi**n / n!`` with Hermitian
    coefficient matrices stored as an array of shape ``(order + 1, D, D)``.
    Unlike :class:`PolynomialPESState`, the constant coefficient ``C_0`` is
    retained because it contains the conditional excitation gaps.
    """

    coefficients: np.ndarray

    def __post_init__(self):
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if self.coefficients.ndim != 3:
            raise ValueError("coefficients must have shape (order + 1, D, D).")
        if self.coefficients.shape[1] != self.coefficients.shape[2]:
            raise ValueError("coefficient matrices must be square.")
        self.coefficients = 0.5 * (
            self.coefficients + np.swapaxes(self.coefficients, 1, 2)
        )

    @property
    def order(self):
        return self.coefficients.shape[0] - 1

    @property
    def dimension(self):
        return self.coefficients.shape[1]

    def asarray(self):
        return np.array(self.coefficients, dtype=float, copy=True)

    @classmethod
    def gaussian(cls, *, order: int, dimension: int, gap: float = 1.0):
        order = int(order)
        dimension = int(dimension)
        if order < 0:
            raise ValueError("order must be non-negative.")
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        coefficients = np.zeros((order + 1, dimension, dimension), dtype=float)
        coefficients[0] = float(gap) * np.diag(np.arange(dimension, dtype=float))
        return cls(coefficients)


@dataclass
class MatrixOneModeNARGShellProjection:
    """One-mode shell projection for a matrix-valued running PES."""

    parameters: OneModeNARGShellParameters
    state: MatrixPolynomialPESState
    grid: np.ndarray
    shell_matrices: np.ndarray
    shell_coefficients: np.ndarray
    residual_norm: float
    reference_energies: np.ndarray

    @property
    def surfaces(self):
        return np.linalg.eigvalsh(self.shell_matrices)


@dataclass
class MatrixOneModeNARGEffectiveHamiltonian:
    """Overlap-dressed active-phi Hamiltonian for matrix analytical NARG.

    This is the Born-Huang realization of the multistate flow: the running
    matrix PES defines conditional shell states ``A_i^a`` on the active
    ``phi`` grid, and their point-to-point overlaps dress the slow kinetic
    operator.
    """

    parameters: OneModeNARGShellParameters
    state: MatrixPolynomialPESState
    phi_grid: np.ndarray
    phi_kinetic: np.ndarray
    hamiltonian: np.ndarray
    conditional_vectors: np.ndarray
    conditional_blocks: np.ndarray
    kinetic_dressing: np.ndarray
    effective_energies: np.ndarray

    @property
    def n_conditional_states(self):
        return self.conditional_vectors.shape[2]

    @property
    def conditional_energies(self):
        return np.linalg.eigvalsh(self.conditional_blocks)


@dataclass
class MatrixOneModeNARGFlow:
    """Integrated matrix-valued one-mode analytical NARG trajectory."""

    parameters: OneModeNARGShellParameters
    l: np.ndarray
    states: np.ndarray
    normalize_gap: bool = True
    gauge_ground: bool = True

    @property
    def coefficients(self):
        return self.states

    @property
    def constant_energies(self):
        return np.linalg.eigvalsh(self.states[:, 0])

    @property
    def ground_energies(self):
        return self.constant_energies[:, 0]

    @property
    def first_gaps(self):
        if self.states.shape[2] < 2:
            return np.zeros(len(self.states), dtype=float)
        energies = self.constant_energies
        return energies[:, 1] - energies[:, 0]

    def state(self, index: int = -1):
        """Return one trajectory point as a matrix polynomial PES state."""
        return MatrixPolynomialPESState(self.states[int(index)])


@dataclass
class MatrixOneModeNARGLinearization:
    """Linearization of the normalized matrix one-mode NARG flow."""

    parameters: OneModeNARGShellParameters
    fixed_point: MatrixPolynomialPESState
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    packed_labels: list[str]


@dataclass
class DiscreteMatrixWilsonNARGStep:
    """One coordinate-coupled Wilson-chain matrix conditional NARG step."""

    site: int
    onsite: float
    coupling: float
    block_scale: float
    shell_scale: float
    coefficients: np.ndarray
    residual_norm: float
    retained_energies: np.ndarray
    normalization_gap: float


@dataclass
class DiscreteMatrixWilsonNARGFlow:
    """Coordinate-coupled Wilson-chain matrix conditional NARG flow.

    This is a PES closure, not the exact bosonic Wilson-chain update.  The
    exact update must propagate the projected boundary annihilation operator
    and add ``t(B^dagger b + B b^dagger)``.
    """

    initial_state: MatrixPolynomialPESState
    steps: list[DiscreteMatrixWilsonNARGStep]
    n_conditional_states: int
    polynomial_order: int
    nrg_rescaled: bool = False
    Lambda: float | None = None
    rescale_power: float = 1.0

    @property
    def coefficients(self):
        return np.asarray([step.coefficients for step in self.steps], dtype=float)

    @property
    def gaps(self):
        return np.asarray([step.normalization_gap for step in self.steps], dtype=float)

    @property
    def residuals(self):
        return np.asarray([step.residual_norm for step in self.steps], dtype=float)


def _default_initial_coupling(chain):
    return float(chain.impurity_coupling) / np.sqrt(2.0)


def _scale_for_site(site, *, nrg_rescale, Lambda, rescale_power):
    if not nrg_rescale:
        return 1.0
    return float(Lambda) ** (float(rescale_power) * int(site))


def analytical_harmonic_narg_flow(
    chain,
    impurity_stiffness: float,
    *,
    initial_coupling: float | None = None,
    nrg_rescale: bool = False,
    Lambda: float | None = None,
    rescale_power: float = 1.0,
    stop_on_unstable: bool = True,
):
    """Run the Gaussian analytical NARG susceptibility recursion.

    The block response is

    ``E_N(F) - E_N(0) = -0.5 * chi_N * F**2``.

    Adding Wilson coordinate ``q`` gives the analytical NARG PES

    ``V(q) = 0.5 * omega_N * q**2 - 0.5 * chi_N * (v_N q)**2``.

    With explicit NRG rescaling, the same convention as
    :class:`SpinBosonWilsonNARG` is used:

    ``K_{N+1} = scale_N H_{N+1}``.
    """
    stiffness = float(impurity_stiffness)
    if stiffness <= 0.0:
        raise ValueError("impurity_stiffness must be positive.")
    if chain.nmodes < 1:
        raise ValueError("chain must contain at least one Wilson mode.")
    if nrg_rescale:
        if Lambda is None:
            raise ValueError("Lambda is required when nrg_rescale=True.")
        if float(Lambda) <= 1.0:
            raise ValueError("Lambda must be larger than one.")

    coupling0 = (
        _default_initial_coupling(chain)
        if initial_coupling is None
        else float(initial_coupling)
    )
    if coupling0 <= 0.0:
        raise ValueError("initial_coupling must be positive.")

    chi = 1.0 / stiffness
    steps = []
    for site, onsite in enumerate(np.asarray(chain.onsite, dtype=float)):
        shell_scale = _scale_for_site(
            site,
            nrg_rescale=nrg_rescale,
            Lambda=Lambda,
            rescale_power=rescale_power,
        )
        if site == 0:
            block_scale = 1.0
            coupling = coupling0
        else:
            previous_scale = _scale_for_site(
                site - 1,
                nrg_rescale=nrg_rescale,
                Lambda=Lambda,
                rescale_power=rescale_power,
            )
            block_scale = shell_scale / previous_scale
            coupling = float(chain.hopping[site - 1])

        curvature = (
            shell_scale * float(onsite)
            - (shell_scale * shell_scale / block_scale) * coupling * coupling * chi
        )
        stable = bool(np.isfinite(curvature) and curvature > 0.0)
        susceptibility = 1.0 / curvature if stable else np.inf
        steps.append(
            AnalyticalHarmonicNARGStep(
                site=int(site),
                onsite=float(onsite),
                coupling=float(coupling),
                input_susceptibility=float(chi),
                curvature=float(curvature),
                susceptibility=float(susceptibility),
                block_scale=float(block_scale),
                shell_scale=float(shell_scale),
                stable=stable,
            )
        )
        if not stable and stop_on_unstable:
            break
        chi = susceptibility

    return AnalyticalHarmonicNARGResult(
        impurity_stiffness=stiffness,
        initial_coupling=coupling0,
        steps=steps,
        nrg_rescaled=bool(nrg_rescale),
        Lambda=None if Lambda is None else float(Lambda),
        rescale_power=float(rescale_power),
    )


def analytical_landau_narg_flow(
    chain,
    impurity_stiffness: float,
    impurity_quartic: float,
    *,
    initial_coupling: float | None = None,
    nrg_rescale: bool = False,
    Lambda: float | None = None,
    rescale_power: float = 1.0,
    stop_on_unstable: bool = True,
):
    """Run the quartic Landau analytical NARG recursion.

    The retained block is represented by

    ``V_N(x) = 0.5 * r_N * x**2 + u_N * x**4 / 24``.

    Its small-force generating function is

    ``E_N(F) = -F**2/(2 r_N) + u_N F**4/(24 r_N**4) + ...``.

    Adding the next Wilson coordinate gives

    ``r' = omega - v**2/r``, ``u' = v**4 u/r**4``,

    with the same explicit NRG rescaling convention used by
    :func:`analytical_harmonic_narg_flow`.
    """
    stiffness = float(impurity_stiffness)
    quartic = float(impurity_quartic)
    if stiffness <= 0.0:
        raise ValueError("impurity_stiffness must be positive.")
    if chain.nmodes < 1:
        raise ValueError("chain must contain at least one Wilson mode.")
    if nrg_rescale:
        if Lambda is None:
            raise ValueError("Lambda is required when nrg_rescale=True.")
        if float(Lambda) <= 1.0:
            raise ValueError("Lambda must be larger than one.")
    coupling0 = (
        _default_initial_coupling(chain)
        if initial_coupling is None
        else float(initial_coupling)
    )
    if coupling0 <= 0.0:
        raise ValueError("initial_coupling must be positive.")

    r_current = stiffness
    u_current = quartic
    steps = []
    for site, onsite in enumerate(np.asarray(chain.onsite, dtype=float)):
        shell_scale = _scale_for_site(
            site,
            nrg_rescale=nrg_rescale,
            Lambda=Lambda,
            rescale_power=rescale_power,
        )
        if site == 0:
            block_scale = 1.0
            coupling = coupling0
        else:
            previous_scale = _scale_for_site(
                site - 1,
                nrg_rescale=nrg_rescale,
                Lambda=Lambda,
                rescale_power=rescale_power,
            )
            block_scale = shell_scale / previous_scale
            coupling = float(chain.hopping[site - 1])

        stable_input = np.isfinite(r_current) and r_current > 0.0
        if stable_input:
            susceptibility = 1.0 / r_current
            quartic_response = -u_current / r_current**4
            curvature = (
                shell_scale * float(onsite)
                - (shell_scale * shell_scale / block_scale)
                * coupling
                * coupling
                * susceptibility
            )
            quartic_factor = (
                shell_scale**4
                / block_scale**3
                * coupling**4
                / r_current**4
            )
            next_quartic = quartic_factor * u_current
            field_scale = shell_scale * coupling * susceptibility
            stable = bool(np.isfinite(curvature) and curvature > 0.0)
        else:
            susceptibility = np.inf
            quartic_response = np.nan
            curvature = np.nan
            quartic_factor = np.nan
            next_quartic = np.nan
            field_scale = np.nan
            stable = False

        steps.append(
            AnalyticalLandauNARGStep(
                site=int(site),
                onsite=float(onsite),
                coupling=float(coupling),
                input_stiffness=float(r_current),
                input_quartic=float(u_current),
                curvature=float(curvature),
                quartic=float(next_quartic),
                susceptibility=float(susceptibility),
                quartic_response=float(quartic_response),
                field_scale=float(field_scale),
                quartic_scale=float(quartic_factor),
                block_scale=float(block_scale),
                shell_scale=float(shell_scale),
                stable=stable,
            )
        )
        if not stable and stop_on_unstable:
            break
        r_current = curvature
        u_current = next_quartic

    return AnalyticalLandauNARGResult(
        impurity_stiffness=stiffness,
        impurity_quartic=quartic,
        initial_coupling=coupling0,
        steps=steps,
        nrg_rescaled=bool(nrg_rescale),
        Lambda=None if Lambda is None else float(Lambda),
        rescale_power=float(rescale_power),
    )


def critical_harmonic_impurity_stiffness(chain, *, initial_coupling: float | None = None):
    """Return the finite-chain Gaussian critical stiffness.

    This is the exact continued-fraction instability of the analytical NARG
    recursion.  For a one-mode bath it reduces to ``c0**2 / omega0``.
    """
    if chain.nmodes < 1:
        raise ValueError("chain must contain at least one Wilson mode.")
    onsite = np.asarray(chain.onsite, dtype=float)
    hopping = np.asarray(chain.hopping, dtype=float)
    coupling0 = (
        _default_initial_coupling(chain)
        if initial_coupling is None
        else float(initial_coupling)
    )
    if coupling0 <= 0.0:
        raise ValueError("initial_coupling must be positive.")

    if len(onsite) == 1:
        critical_chi = onsite[0] / (coupling0 * coupling0)
    else:
        allowed_chi = onsite[-1] / (hopping[-1] * hopping[-1])
        for site in range(len(onsite) - 2, 0, -1):
            allowed_chi = (onsite[site] - 1.0 / allowed_chi) / (
                hopping[site - 1] * hopping[site - 1]
            )
            if not np.isfinite(allowed_chi) or allowed_chi <= 0.0:
                raise ValueError("Wilson chain is not positive in the harmonic recursion.")
        critical_chi = (onsite[0] - 1.0 / allowed_chi) / (coupling0 * coupling0)

    if not np.isfinite(critical_chi) or critical_chi <= 0.0:
        raise ValueError("critical susceptibility is not positive.")
    return float(1.0 / critical_chi)


def stationary_curvature_window(
    curvatures,
    *,
    window: int = 8,
    min_start: int = 3,
    exclude_tail: int = 2,
):
    """Return the most stationary contiguous curvature window."""
    values = np.asarray(curvatures, dtype=float)
    if values.ndim != 1:
        raise ValueError("curvatures must be one-dimensional.")
    window = int(window)
    min_start = int(min_start)
    exclude_tail = int(exclude_tail)
    if window < 1:
        raise ValueError("window must be positive.")
    stop_limit = len(values) - max(0, exclude_tail)
    if stop_limit - min_start < window:
        return np.array([], dtype=int)

    best_score = np.inf
    best_start = None
    for start in range(max(0, min_start), stop_limit - window + 1):
        segment = values[start : start + window]
        if not np.all(np.isfinite(segment)) or np.any(segment <= 0.0):
            continue
        mean = float(np.mean(segment))
        if mean <= 0.0:
            continue
        score = float(np.std(segment) / mean)
        if score < best_score:
            best_score = score
            best_start = start
    if best_start is None:
        return np.array([], dtype=int)
    return np.arange(best_start, best_start + window, dtype=int)


def estimate_harmonic_thermal_exponent(
    chain,
    stiffness: float,
    *,
    relative_step: float = 1e-2,
    initial_coupling: float | None = None,
    Lambda: float,
    rescale_power: float = 1.0,
    tail: int = 8,
):
    """Estimate the relevant exponent from a stiffness perturbation.

    The calculation is still analytical: two closed susceptibility recursions
    are compared, and the local exponent is

    ``log(|delta kappa_{N+1}| / |delta kappa_N|) / log(Lambda)``.
    """
    if float(relative_step) <= 0.0:
        raise ValueError("relative_step must be positive.")
    reference = analytical_harmonic_narg_flow(
        chain,
        stiffness,
        initial_coupling=initial_coupling,
        nrg_rescale=True,
        Lambda=Lambda,
        rescale_power=rescale_power,
    )
    perturbed_stiffness = float(stiffness) * (1.0 + float(relative_step))
    perturbed = analytical_harmonic_narg_flow(
        chain,
        perturbed_stiffness,
        initial_coupling=initial_coupling,
        nrg_rescale=True,
        Lambda=Lambda,
        rescale_power=rescale_power,
    )
    nsteps = min(len(reference.steps), len(perturbed.steps))
    delta = np.abs(perturbed.curvatures[:nsteps] - reference.curvatures[:nsteps])
    local_y = np.full(max(0, nsteps - 1), np.nan, dtype=float)
    for site in range(len(local_y)):
        if delta[site] > 1e-14 and delta[site + 1] > 1e-14:
            local_y[site] = np.log(delta[site + 1] / delta[site]) / np.log(float(Lambda))

    finite = np.flatnonzero(np.isfinite(local_y))
    window = finite[-int(tail) :] if len(finite) else np.array([], dtype=int)
    if len(window):
        mean_y = float(np.mean(local_y[window]))
        std_y = float(np.std(local_y[window]))
    else:
        mean_y = np.nan
        std_y = np.nan
    return AnalyticalHarmonicExponentEstimate(
        stiffness=float(stiffness),
        perturbed_stiffness=perturbed_stiffness,
        local_y=local_y,
        mean_y=mean_y,
        std_y=std_y,
        window=np.asarray(window, dtype=int),
    )


def _coerce_continuum_parameters(parameters=None, **kwargs):
    if parameters is not None and kwargs:
        raise ValueError("pass either parameters or keyword parameters, not both.")
    if parameters is not None:
        if isinstance(parameters, ContinuumAnalyticalNARGParameters):
            return parameters
        raise TypeError("parameters must be a ContinuumAnalyticalNARGParameters instance.")
    return ContinuumAnalyticalNARGParameters(**kwargs)


def continuum_landau_beta(state, parameters=None, **kwargs):
    """Continuum analytical-NARG beta function.

    The coordinates are already centered on the critical surface, so the
    thermal equation has no additive tadpole term:

    ``dt/dl = (s + a u) t``.

    The quartic flow includes the marginally irrelevant correction at
    ``s = 1/2``:

    ``du/dl = (2s - 1)u - b u**2``.
    """
    params = _coerce_continuum_parameters(parameters, **kwargs)
    if isinstance(state, ContinuumAnalyticalNARGState):
        thermal, quartic, field = state.asarray()
    else:
        thermal, quartic, field = np.asarray(state, dtype=float)
    y_t = params.s
    y_u = 2.0 * params.s - 1.0
    y_h = 0.5 * (1.0 + params.s)
    return np.array(
        [
            (y_t + params.thermal_quartic * quartic) * thermal,
            y_u * quartic - params.quartic_self * quartic * quartic,
            (y_h + params.field_quartic * quartic) * field,
        ],
        dtype=float,
    )


def continuum_landau_fixed_point(parameters=None, *, kind: str = "gaussian", **kwargs):
    """Return a continuum aNARG fixed point."""
    params = _coerce_continuum_parameters(parameters, **kwargs)
    key = str(kind).lower()
    y_u = 2.0 * params.s - 1.0
    if key in {"gaussian", "g"}:
        quartic = 0.0
    elif key in {"interacting", "wilson-fisher", "wf"}:
        if params.quartic_self <= 0.0:
            raise ValueError("interacting fixed point requires quartic_self > 0.")
        if y_u <= 0.0:
            raise ValueError("interacting fixed point exists only when 2s - 1 > 0.")
        quartic = y_u / params.quartic_self
    elif key == "auto":
        quartic = y_u / params.quartic_self if y_u > 0.0 and params.quartic_self > 0.0 else 0.0
    else:
        raise ValueError("kind must be 'gaussian', 'interacting', or 'auto'.")
    return ContinuumAnalyticalNARGState(thermal=0.0, quartic=float(quartic), field=0.0)


def continuum_landau_jacobian(state, parameters=None, **kwargs):
    """Return the Jacobian of the continuum aNARG beta function."""
    params = _coerce_continuum_parameters(parameters, **kwargs)
    thermal, quartic, field = (
        state.asarray()
        if isinstance(state, ContinuumAnalyticalNARGState)
        else np.asarray(state, dtype=float)
    )
    y_t = params.s
    y_u = 2.0 * params.s - 1.0
    y_h = 0.5 * (1.0 + params.s)
    return np.array(
        [
            [y_t + params.thermal_quartic * quartic, params.thermal_quartic * thermal, 0.0],
            [0.0, y_u - 2.0 * params.quartic_self * quartic, 0.0],
            [0.0, params.field_quartic * field, y_h + params.field_quartic * quartic],
        ],
        dtype=float,
    )


def continuum_landau_linearization(parameters=None, *, kind: str = "gaussian", **kwargs):
    """Linearize the continuum analytical-NARG flow at a fixed point."""
    params = _coerce_continuum_parameters(parameters, **kwargs)
    fixed_point = continuum_landau_fixed_point(params, kind=kind)
    jacobian = continuum_landau_jacobian(fixed_point, params)
    eigenvalues = np.linalg.eigvals(jacobian)
    if np.max(np.abs(np.imag(eigenvalues))) <= 1e-12:
        eigenvalues = np.real(eigenvalues)
    return ContinuumAnalyticalNARGLinearization(
        parameters=params,
        fixed_point=fixed_point,
        jacobian=jacobian,
        eigenvalues=np.asarray(eigenvalues),
        exponents=landau_critical_exponents(params.s),
    )


def integrate_continuum_landau_flow(
    initial_state,
    parameters=None,
    *,
    lmax: float = 8.0,
    nsteps: int = 400,
    **kwargs,
):
    """Integrate the continuum analytical-NARG beta functions with RK4."""
    params = _coerce_continuum_parameters(parameters, **kwargs)
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("nsteps must be positive.")
    lmax = float(lmax)
    if lmax <= 0.0:
        raise ValueError("lmax must be positive.")
    state = (
        initial_state.asarray()
        if isinstance(initial_state, ContinuumAnalyticalNARGState)
        else np.asarray(initial_state, dtype=float)
    )
    if state.shape != (3,):
        raise ValueError("initial_state must contain thermal, quartic, and field.")

    grid = np.linspace(0.0, lmax, nsteps + 1)
    states = np.empty((nsteps + 1, 3), dtype=float)
    states[0] = state
    step = grid[1] - grid[0]
    for index in range(nsteps):
        current = states[index]
        k1 = continuum_landau_beta(current, params)
        k2 = continuum_landau_beta(current + 0.5 * step * k1, params)
        k3 = continuum_landau_beta(current + 0.5 * step * k2, params)
        k4 = continuum_landau_beta(current + step * k3, params)
        states[index + 1] = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return ContinuumAnalyticalNARGFlow(parameters=params, l=grid, states=states)


def _coerce_ck_parameters(parameters=None, **kwargs):
    if parameters is not None and kwargs:
        raise ValueError("pass either parameters or keyword parameters, not both.")
    if parameters is not None:
        if isinstance(parameters, CKLocalPotentialParameters):
            return parameters
        raise TypeError("parameters must be a CKLocalPotentialParameters instance.")
    return CKLocalPotentialParameters(**kwargs)


def ck_local_potential_beta(state, parameters=None, **kwargs):
    """Quartic CK local-potential beta function.

    This is the polynomial projection of

    ``d_l U = U - d_phi phi U' + A log(1 + U'')``

    onto ``U = -h phi + r phi**2/2 + u phi**4/24``.
    """
    params = _coerce_ck_parameters(parameters, **kwargs)
    mass, quartic, field = (
        state.asarray()
        if isinstance(state, CKLocalPotentialState)
        else np.asarray(state, dtype=float)
    )
    denom = 1.0 + mass
    if abs(denom) <= 1e-14:
        raise ValueError("local-potential flow is singular at mass = -1.")
    a = params.shell_measure
    return np.array(
        [
            params.s * mass + a * quartic / denom,
            (2.0 * params.s - 1.0) * quartic
            - 3.0 * a * quartic * quartic / (denom * denom),
            0.5 * (1.0 + params.s) * field,
        ],
        dtype=float,
    )


def ck_local_potential_fixed_point(parameters=None, *, kind: str = "gaussian", **kwargs):
    """Return a fixed point of the quartic CK local-potential flow."""
    params = _coerce_ck_parameters(parameters, **kwargs)
    key = str(kind).lower()
    y_u = 2.0 * params.s - 1.0
    if key in {"gaussian", "g"}:
        mass = 0.0
        quartic = 0.0
    elif key in {"interacting", "wilson-fisher", "wf"}:
        if y_u <= 0.0:
            raise ValueError("interacting fixed point exists only when 2s - 1 > 0.")
        denom = 5.0 * params.s - 1.0
        if abs(denom) <= 1e-14 or params.shell_measure <= 0.0:
            raise ValueError("interacting fixed point is singular for these parameters.")
        mass = -y_u / denom
        quartic = -params.s * mass * (1.0 + mass) / params.shell_measure
    elif key == "auto":
        if y_u > 0.0:
            return ck_local_potential_fixed_point(params, kind="interacting")
        return ck_local_potential_fixed_point(params, kind="gaussian")
    else:
        raise ValueError("kind must be 'gaussian', 'interacting', or 'auto'.")
    return CKLocalPotentialState(mass=float(mass), quartic=float(quartic), field=0.0)


def ck_local_potential_jacobian(state, parameters=None, **kwargs):
    """Return the Jacobian of the quartic CK local-potential beta function."""
    params = _coerce_ck_parameters(parameters, **kwargs)
    mass, quartic, _field = (
        state.asarray()
        if isinstance(state, CKLocalPotentialState)
        else np.asarray(state, dtype=float)
    )
    denom = 1.0 + mass
    if abs(denom) <= 1e-14:
        raise ValueError("local-potential flow is singular at mass = -1.")
    a = params.shell_measure
    return np.array(
        [
            [params.s - a * quartic / (denom * denom), a / denom, 0.0],
            [
                6.0 * a * quartic * quartic / (denom**3),
                2.0 * params.s - 1.0 - 6.0 * a * quartic / (denom * denom),
                0.0,
            ],
            [0.0, 0.0, 0.5 * (1.0 + params.s)],
        ],
        dtype=float,
    )


def ck_local_potential_linearization(parameters=None, *, kind: str = "gaussian", **kwargs):
    """Linearize the CK local-potential beta function at a fixed point."""
    params = _coerce_ck_parameters(parameters, **kwargs)
    fixed_point = ck_local_potential_fixed_point(params, kind=kind)
    jacobian = ck_local_potential_jacobian(fixed_point, params)
    eigenvalues = np.linalg.eigvals(jacobian)
    if np.max(np.abs(np.imag(eigenvalues))) <= 1e-12:
        eigenvalues = np.real(eigenvalues)
    return CKLocalPotentialLinearization(
        parameters=params,
        fixed_point=fixed_point,
        jacobian=jacobian,
        eigenvalues=np.asarray(eigenvalues),
    )


def integrate_ck_local_potential_flow(
    initial_state,
    parameters=None,
    *,
    lmax: float = 8.0,
    nsteps: int = 400,
    **kwargs,
):
    """Integrate the CK quartic local-potential ODEs with RK4."""
    params = _coerce_ck_parameters(parameters, **kwargs)
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("nsteps must be positive.")
    lmax = float(lmax)
    if lmax <= 0.0:
        raise ValueError("lmax must be positive.")
    state = (
        initial_state.asarray()
        if isinstance(initial_state, CKLocalPotentialState)
        else np.asarray(initial_state, dtype=float)
    )
    if state.shape != (3,):
        raise ValueError("initial_state must contain mass, quartic, and field.")

    grid = np.linspace(0.0, lmax, nsteps + 1)
    states = np.empty((nsteps + 1, 3), dtype=float)
    states[0] = state
    step = grid[1] - grid[0]
    for index in range(nsteps):
        current = states[index]
        k1 = ck_local_potential_beta(current, params)
        k2 = ck_local_potential_beta(current + 0.5 * step * k1, params)
        k3 = ck_local_potential_beta(current + 0.5 * step * k2, params)
        k4 = ck_local_potential_beta(current + step * k3, params)
        states[index + 1] = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return CKLocalPotentialFlow(parameters=params, l=grid, states=states)


def _coerce_one_mode_parameters(parameters=None, **kwargs):
    if parameters is not None and kwargs:
        raise ValueError("pass either parameters or keyword parameters, not both.")
    if parameters is not None:
        if isinstance(parameters, OneModeNARGShellParameters):
            return parameters
        raise TypeError("parameters must be a OneModeNARGShellParameters instance.")
    return OneModeNARGShellParameters(**kwargs)


def _coerce_ck_state(state):
    if isinstance(state, CKLocalPotentialState):
        mass, quartic, field = state.asarray()
    else:
        mass, quartic, field = np.asarray(state, dtype=float)
    return CKLocalPotentialState(
        mass=float(mass),
        quartic=float(quartic),
        field=float(field),
    )


def _fock_coordinate_and_number(dim: int):
    lowering = np.zeros((dim, dim), dtype=float)
    for n in range(1, dim):
        lowering[n - 1, n] = np.sqrt(n)
    raising = lowering.T
    coordinate = (lowering + raising) / np.sqrt(2.0)
    number = raising @ lowering
    return coordinate, number


def _fock_coordinate_and_kinetic(dim: int):
    coordinate, number = _fock_coordinate_and_number(dim)
    kinetic = number + 0.5 * np.eye(dim, dtype=float) - 0.5 * (coordinate @ coordinate)
    return coordinate, 0.5 * (kinetic + kinetic.T)


def harmonic_oscillator_displacement_overlap(displacement: float, nstates: int):
    """Return ``<m|D(displacement)|n>`` for real harmonic displacements.

    This is the analytic Franck-Condon overlap between oscillator states whose
    displacement parameters differ by ``displacement``.
    """
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")
    alpha = float(displacement)
    overlap = np.empty((nstates, nstates), dtype=float)
    prefactor = np.exp(-0.5 * alpha * alpha)
    factorials = [factorial(level) for level in range(nstates)]
    for m in range(nstates):
        for n in range(nstates):
            total = 0.0
            start = max(0, n - m)
            for j in range(start, n + 1):
                ell = m - n + j
                total += (
                    ((-alpha) ** j)
                    * (alpha**ell)
                    / (factorial(j) * factorial(ell) * factorial(n - j))
                )
            overlap[m, n] = (
                prefactor
                * np.sqrt(float(factorials[m]) * float(factorials[n]))
                * total
            )
    return overlap


def _quartic_potential_value(state: CKLocalPotentialState, phi):
    phi = np.asarray(phi, dtype=float)
    return (
        -state.field * phi
        + 0.5 * state.mass * phi * phi
        + state.quartic * phi**4 / 24.0
    )


def _quartic_potential_matrix(state: CKLocalPotentialState, coordinate):
    coordinate2 = coordinate @ coordinate
    coordinate4 = coordinate2 @ coordinate2
    return (
        -state.field * coordinate
        + 0.5 * state.mass * coordinate2
        + state.quartic * coordinate4 / 24.0
    )


def one_mode_narg_shell_energy(phi, state, parameters=None, **kwargs):
    """Return the one-mode eigenvalue shell correction.

    For a running potential ``U(phi)``, the shell functional is

    ``A [E0(phi) - E0(0) - U(phi)]``,

    where ``E0(phi)`` is the ground-state energy of

    ``omega b^dagger b + U(phi + a q)``.

    The subtraction removes the tree-level potential already accounted for by
    the scaling part of the beta function.
    """
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_ck_state(state)
    points = np.asarray(phi, dtype=float)
    flat_points = points.reshape(-1)
    dim = params.basis_size
    identity = np.eye(dim, dtype=float)
    coordinate, number = _fock_coordinate_and_number(dim)
    shell_coordinate = params.fluctuation_amplitude * coordinate
    bare = params.oscillator_frequency * number

    def ground_energy(background):
        shifted = float(background) * identity + shell_coordinate
        hamiltonian = bare + _quartic_potential_matrix(current, shifted)
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
        return float(np.linalg.eigvalsh(hamiltonian)[0])

    reference = ground_energy(0.0)
    values = np.empty_like(flat_points, dtype=float)
    for index, point in enumerate(flat_points):
        values[index] = (
            ground_energy(point)
            - reference
            - float(_quartic_potential_value(current, point))
        )
    values *= params.shell_measure
    return values.reshape(points.shape)


def one_mode_narg_shell_projection(state, parameters=None, **kwargs):
    """Project the one-mode shell energy onto ``h, r, u`` beta terms."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_ck_state(state)
    grid = np.linspace(-params.fit_radius, params.fit_radius, params.n_fit_points)
    shell = one_mode_narg_shell_energy(grid, current, params)
    basis = np.column_stack([grid, grid**2, grid**3, grid**4])
    coefficients, residuals, _, _ = np.linalg.lstsq(basis, shell, rcond=None)
    fitted = basis @ coefficients
    if residuals.size:
        residual_norm = float(np.sqrt(residuals[0]))
    else:
        residual_norm = float(np.linalg.norm(shell - fitted))
    beta_shell = np.array(
        [
            2.0 * coefficients[1],
            24.0 * coefficients[3],
            -coefficients[0],
        ],
        dtype=float,
    )
    return OneModeNARGShellProjection(
        parameters=params,
        state=current,
        grid=grid,
        shell_energy=shell,
        polynomial_coefficients=np.asarray(coefficients, dtype=float),
        beta_shell=beta_shell,
        residual_norm=residual_norm,
    )


def one_mode_narg_beta(state, parameters=None, **kwargs):
    """NARG beta function with an explicit one-mode eigenvalue shell."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_ck_state(state)
    scaling = np.array(
        [
            params.s * current.mass,
            (2.0 * params.s - 1.0) * current.quartic,
            0.5 * (1.0 + params.s) * current.field,
        ],
        dtype=float,
    )
    projection = one_mode_narg_shell_projection(current, params)
    return scaling + projection.beta_shell


def one_mode_narg_fixed_point(parameters=None, *, kind: str = "gaussian", **kwargs):
    """Return a fixed point of the one-mode eigenvalue NARG beta flow."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    key = str(kind).lower()
    if key in {"gaussian", "g", "auto"}:
        return CKLocalPotentialState(mass=0.0, quartic=0.0, field=0.0)
    raise ValueError("only the Gaussian one-mode NARG fixed point is implemented.")


def one_mode_narg_jacobian(
    state,
    parameters=None,
    *,
    step: float = 1e-5,
    **kwargs,
):
    """Finite-difference Jacobian of the one-mode NARG beta function."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    center = _coerce_ck_state(state).asarray()
    step = float(step)
    if step <= 0.0:
        raise ValueError("step must be positive.")
    jacobian = np.empty((3, 3), dtype=float)
    for column in range(3):
        delta = np.zeros(3, dtype=float)
        width = step * max(1.0, abs(center[column]))
        delta[column] = width
        plus = one_mode_narg_beta(center + delta, params)
        minus = one_mode_narg_beta(center - delta, params)
        jacobian[:, column] = (plus - minus) / (2.0 * width)
    return jacobian


def one_mode_narg_linearization(parameters=None, *, kind: str = "gaussian", **kwargs):
    """Linearize the one-mode eigenvalue NARG flow at a fixed point."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    fixed_point = one_mode_narg_fixed_point(params, kind=kind)
    jacobian = one_mode_narg_jacobian(fixed_point, params)
    eigenvalues = np.linalg.eigvals(jacobian)
    if np.max(np.abs(np.imag(eigenvalues))) <= 1e-10:
        eigenvalues = np.real(eigenvalues)
    return OneModeNARGLinearization(
        parameters=params,
        fixed_point=fixed_point,
        jacobian=jacobian,
        eigenvalues=np.asarray(eigenvalues),
        exponents=landau_critical_exponents(params.s),
    )


def integrate_one_mode_narg_flow(
    initial_state,
    parameters=None,
    *,
    lmax: float = 8.0,
    nsteps: int = 100,
    **kwargs,
):
    """Integrate the one-mode eigenvalue NARG beta functions with RK4."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("nsteps must be positive.")
    lmax = float(lmax)
    if lmax <= 0.0:
        raise ValueError("lmax must be positive.")
    state = _coerce_ck_state(initial_state).asarray()

    grid = np.linspace(0.0, lmax, nsteps + 1)
    states = np.empty((nsteps + 1, 3), dtype=float)
    states[0] = state
    step = grid[1] - grid[0]
    for index in range(nsteps):
        current = states[index]
        k1 = one_mode_narg_beta(current, params)
        k2 = one_mode_narg_beta(current + 0.5 * step * k1, params)
        k3 = one_mode_narg_beta(current + 0.5 * step * k2, params)
        k4 = one_mode_narg_beta(current + step * k3, params)
        states[index + 1] = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return OneModeNARGFlow(parameters=params, l=grid, states=states)


def _coerce_polynomial_state(state):
    if isinstance(state, PolynomialPESState):
        return state
    return PolynomialPESState(np.asarray(state, dtype=float))


def polynomial_pes_value(state, phi):
    """Evaluate a Taylor-projected PES."""
    current = _coerce_polynomial_state(state)
    points = np.asarray(phi, dtype=float)
    value = np.zeros_like(points, dtype=float)
    for index, coupling in enumerate(current.couplings, start=1):
        value = value + coupling * points**index / factorial(index)
    return value


def _polynomial_pes_matrix(state, coordinate):
    current = _coerce_polynomial_state(state)
    dim = coordinate.shape[0]
    identity = np.eye(dim, dtype=float)
    power = identity
    potential = np.zeros_like(coordinate, dtype=float)
    for index, coupling in enumerate(current.couplings, start=1):
        power = power @ coordinate
        potential = potential + coupling * power / factorial(index)
    return 0.5 * (potential + potential.T)


def polynomial_one_mode_narg_shell_energy(phi, state, parameters=None, **kwargs):
    """Return the higher-order one-mode NARG shell energy."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_polynomial_state(state)
    points = np.asarray(phi, dtype=float)
    flat_points = points.reshape(-1)
    dim = params.basis_size
    identity = np.eye(dim, dtype=float)
    coordinate, number = _fock_coordinate_and_number(dim)
    shell_coordinate = params.fluctuation_amplitude * coordinate
    bare = params.oscillator_frequency * number

    def ground_energy(background):
        shifted = float(background) * identity + shell_coordinate
        hamiltonian = bare + _polynomial_pes_matrix(current, shifted)
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
        return float(np.linalg.eigvalsh(hamiltonian)[0])

    reference = ground_energy(0.0)
    values = np.empty_like(flat_points, dtype=float)
    for index, point in enumerate(flat_points):
        values[index] = (
            ground_energy(point)
            - reference
            - float(polynomial_pes_value(current, point))
        )
    values *= params.shell_measure
    return values.reshape(points.shape)


def _polynomial_one_mode_hamiltonian(background, state, params):
    current = _coerce_polynomial_state(state)
    dim = params.basis_size
    identity = np.eye(dim, dtype=float)
    coordinate, number = _fock_coordinate_and_number(dim)
    shell_coordinate = float(background) * identity + params.fluctuation_amplitude * coordinate
    hamiltonian = params.oscillator_frequency * number + _polynomial_pes_matrix(
        current,
        shell_coordinate,
    )
    return 0.5 * (hamiltonian + hamiltonian.T)


def conditional_one_mode_narg_shell_projection(
    state,
    parameters=None,
    *,
    n_conditional_states: int = 2,
    order: int | None = None,
    reference_phi: float = 0.0,
    **kwargs,
):
    """Project a one-mode shell onto several retained conditional states.

    The retained basis is the lowest ``n_conditional_states`` eigenvectors of
    the shell Hamiltonian at ``reference_phi``.  The returned object contains
    the matrix PES

    ``A [V^T H(phi) V - E0(reference_phi) I - U(phi) I]``.

    Its constant part stores the retained excitation gaps, while the phi
    dependence stores the matrix-valued shell correction.  Diagonalizing the
    matrix at each phi gives the retained conditional surfaces.
    """
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_polynomial_state(state)
    n_conditional_states = int(n_conditional_states)
    if n_conditional_states < 1:
        raise ValueError("n_conditional_states must be positive.")
    if n_conditional_states > params.basis_size:
        raise ValueError("n_conditional_states cannot exceed basis_size.")
    fit_order = current.order if order is None else int(order)
    if fit_order < 0:
        raise ValueError("order must be non-negative.")

    reference_hamiltonian = _polynomial_one_mode_hamiltonian(
        reference_phi,
        current,
        params,
    )
    reference_energies, reference_vectors = np.linalg.eigh(reference_hamiltonian)
    retained = reference_vectors[:, :n_conditional_states]
    reference_ground = float(reference_energies[0])
    identity_retained = np.eye(n_conditional_states, dtype=float)

    grid = np.linspace(-params.fit_radius, params.fit_radius, params.n_fit_points)
    shell_matrices = np.empty(
        (len(grid), n_conditional_states, n_conditional_states),
        dtype=float,
    )
    for index, point in enumerate(grid):
        hamiltonian = _polynomial_one_mode_hamiltonian(point, current, params)
        projected = retained.T @ hamiltonian @ retained
        shell_matrices[index] = params.shell_measure * (
            projected
            - reference_ground * identity_retained
            - float(polynomial_pes_value(current, point)) * identity_retained
        )
        shell_matrices[index] = 0.5 * (shell_matrices[index] + shell_matrices[index].T)

    basis = np.column_stack(
        [grid**index / factorial(index) for index in range(fit_order + 1)]
    )
    targets = shell_matrices.reshape(len(grid), -1)
    coefficients, residuals, _, _ = np.linalg.lstsq(basis, targets, rcond=None)
    fitted = basis @ coefficients
    if residuals.size:
        residual_norm = float(np.sqrt(np.sum(residuals)))
    else:
        residual_norm = float(np.linalg.norm(targets - fitted))
    shell_couplings = coefficients.reshape(
        fit_order + 1,
        n_conditional_states,
        n_conditional_states,
    )
    shell_couplings = 0.5 * (
        shell_couplings + np.swapaxes(shell_couplings, 1, 2)
    )
    return ConditionalOneModeNARGProjection(
        parameters=params,
        state=current,
        n_conditional_states=n_conditional_states,
        reference_phi=float(reference_phi),
        reference_energies=np.asarray(
            reference_energies[:n_conditional_states],
            dtype=float,
        ),
        grid=grid,
        shell_matrices=shell_matrices,
        shell_couplings=shell_couplings,
        residual_norm=residual_norm,
    )


def _coerce_matrix_polynomial_state(state):
    if isinstance(state, MatrixPolynomialPESState):
        return state
    return MatrixPolynomialPESState(np.asarray(state, dtype=float))


def matrix_polynomial_pes_value(state, phi):
    """Evaluate a matrix-valued Taylor PES at scalar ``phi``."""
    current = _coerce_matrix_polynomial_state(state)
    value = np.zeros((current.dimension, current.dimension), dtype=float)
    for order, matrix in enumerate(current.coefficients):
        value = value + matrix * float(phi) ** order / factorial(order)
    return 0.5 * (value + value.T)


def _matrix_polynomial_pes_operator(state, coordinate):
    current = _coerce_matrix_polynomial_state(state)
    osc_dim = coordinate.shape[0]
    power = np.eye(osc_dim, dtype=float)
    operator = np.zeros(
        (osc_dim * current.dimension, osc_dim * current.dimension),
        dtype=float,
    )
    for order, matrix in enumerate(current.coefficients):
        if order > 0:
            power = power @ coordinate
        operator = operator + np.kron(power / factorial(order), matrix)
    return 0.5 * (operator + operator.T)


def _matrix_one_mode_hamiltonian(background, state, params):
    current = _coerce_matrix_polynomial_state(state)
    coordinate, number = _fock_coordinate_and_number(params.basis_size)
    shifted = float(background) * np.eye(params.basis_size) + params.fluctuation_amplitude * coordinate
    hamiltonian = np.kron(
        params.oscillator_frequency * number,
        np.eye(current.dimension),
    ) + _matrix_polynomial_pes_operator(current, shifted)
    return 0.5 * (hamiltonian + hamiltonian.T)


def _align_conditional_subspace(previous, current):
    overlap = previous.T @ current
    left, _, right_h = np.linalg.svd(overlap, full_matrices=False)
    rotation = right_h.T @ left.T
    return current @ rotation


def matrix_one_mode_narg_conditional_basis(
    state,
    phi_grid,
    parameters=None,
    *,
    n_conditional_states: int | None = None,
    align: bool = True,
    **kwargs,
):
    """Return moving conditional shell states over the active ``phi`` grid.

    The basis at each grid point is formed from the lowest eigenvectors of the
    matrix shell Hamiltonian ``H_shell(phi)``.  Neighboring subspaces are
    Procrustes-aligned by state overlap when ``align=True``.
    """
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_matrix_polynomial_state(state)
    grid = np.asarray(phi_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("phi_grid must be a non-empty one-dimensional array.")
    env_dim = params.basis_size * current.dimension
    nstates = current.dimension if n_conditional_states is None else int(n_conditional_states)
    if nstates < 1 or nstates > env_dim:
        raise ValueError("n_conditional_states must be between 1 and the shell Hilbert dimension.")

    vectors = np.empty((grid.size, env_dim, nstates), dtype=float)
    blocks = np.empty((grid.size, nstates, nstates), dtype=float)
    previous = None
    for index, point in enumerate(grid):
        hamiltonian = _matrix_one_mode_hamiltonian(point, current, params)
        _energies, local_vectors = np.linalg.eigh(hamiltonian)
        retained = local_vectors[:, :nstates]
        if align and previous is not None:
            retained = _align_conditional_subspace(previous, retained)
        vectors[index] = retained
        blocks[index] = retained.T @ hamiltonian @ retained
        blocks[index] = 0.5 * (blocks[index] + blocks[index].T)
        previous = retained
    return vectors, blocks


def _overlap_tensor_from_callable(phi_grid, nstates, overlap):
    grid = np.asarray(phi_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("phi_grid must be one-dimensional when overlap is callable.")
    dressing = None
    for i, phi_i in enumerate(grid):
        for j, phi_j in enumerate(grid):
            block = np.asarray(overlap(float(phi_i), float(phi_j)))
            if block.shape != (nstates, nstates):
                raise ValueError("overlap callable must return shape (nstates, nstates).")
            if dressing is None:
                dressing = np.empty(
                    (grid.size, nstates, grid.size, nstates),
                    dtype=block.dtype,
                )
            dressing[i, :, j, :] = block
    return dressing


def _coerce_overlap_tensor(overlap, *, phi_grid, nphi, nstates):
    if callable(overlap):
        return _overlap_tensor_from_callable(phi_grid, nstates, overlap)
    array = np.asarray(overlap)
    if array.shape == (nphi, nstates, nphi, nstates):
        return array
    if array.shape == (nphi, nphi, nstates, nstates):
        return np.swapaxes(array, 1, 2)
    raise ValueError(
        "overlap must have shape (nphi, nstates, nphi, nstates), "
        "(nphi, nphi, nstates, nstates), or be a callable."
    )


def overlap_dressed_phi_kinetic(
    phi_kinetic,
    conditional_vectors=None,
    *,
    phi_grid=None,
    overlap=None,
    nstates: int | None = None,
):
    """Dress active-coordinate kinetic energy by conditional-state overlaps.

    ``overlap`` may be an analytic tensor/callable for
    ``<A(phi_i)^a|A(phi_j)^b>``.  If it is omitted, overlaps are computed
    directly from ``conditional_vectors``.
    """
    kinetic = np.asarray(phi_kinetic)
    if kinetic.ndim != 2 or kinetic.shape[0] != kinetic.shape[1]:
        raise ValueError("phi_kinetic must be a square matrix.")
    if overlap is not None:
        if nstates is None:
            if conditional_vectors is None:
                raise ValueError("nstates is required when overlap is supplied without conditional_vectors.")
            nstates = np.asarray(conditional_vectors).shape[2]
        nphi = kinetic.shape[0]
        dressing = _coerce_overlap_tensor(
            overlap,
            phi_grid=phi_grid,
            nphi=nphi,
            nstates=int(nstates),
        )
    else:
        vectors = np.asarray(conditional_vectors)
        if vectors.ndim != 3:
            raise ValueError("conditional_vectors must have shape (nphi, env_dim, nstates).")
        if kinetic.shape[0] != vectors.shape[0]:
            raise ValueError("phi_kinetic dimension must match the phi grid size.")
        dressing = np.einsum("pia,qib->paqb", vectors.conj(), vectors, optimize=True)
        nstates = vectors.shape[2]
    hamiltonian = np.einsum("pq,paqb->paqb", kinetic, dressing, optimize=True)
    nphi = kinetic.shape[0]
    return (
        hamiltonian.reshape(nphi * nstates, nphi * nstates),
        dressing,
    )


def matrix_one_mode_narg_effective_hamiltonian(
    state,
    phi_grid,
    phi_kinetic,
    parameters=None,
    *,
    n_conditional_states: int | None = None,
    overlap=None,
    align: bool = True,
    **kwargs,
):
    """Build the full nonadiabatic active-``phi`` analytical NARG Hamiltonian.

    The kinetic term is projected through point-to-point overlaps of the moving
    conditional shell states:

    ``H_kin[(i,a),(j,b)] = T_phi[i,j] <A_i^a | A_j^b>``.

    The full matrix assembled here is

    ``H_NARG[(i,a),(j,b)] = H_kin[(i,a),(j,b)]
       + delta_ij <A_i^a|H_shell(phi_i)|A_i^b>``.
    """
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_matrix_polynomial_state(state)
    grid = np.asarray(phi_grid, dtype=float)
    kinetic = np.asarray(phi_kinetic, dtype=float)
    if kinetic.shape != (grid.size, grid.size):
        raise ValueError("phi_kinetic shape must be (len(phi_grid), len(phi_grid)).")

    vectors, blocks = matrix_one_mode_narg_conditional_basis(
        current,
        grid,
        params,
        n_conditional_states=n_conditional_states,
        align=align,
    )
    hamiltonian, dressing = overlap_dressed_phi_kinetic(
        kinetic,
        vectors,
        phi_grid=grid,
        overlap=overlap,
    )
    nstates = vectors.shape[2]
    for index, block in enumerate(blocks):
        rows = slice(index * nstates, (index + 1) * nstates)
        hamiltonian[rows, rows] += block
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T.conj())
    return MatrixOneModeNARGEffectiveHamiltonian(
        parameters=params,
        state=current,
        phi_grid=grid,
        phi_kinetic=kinetic,
        hamiltonian=hamiltonian,
        conditional_vectors=vectors,
        conditional_blocks=blocks,
        kinetic_dressing=dressing,
        effective_energies=np.linalg.eigvalsh(hamiltonian),
    )


def matrix_one_mode_narg_shell_projection(state, parameters=None, **kwargs):
    """Project the one-mode NARG shell for a matrix-valued PES."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_matrix_polynomial_state(state)
    dim = current.dimension
    reference_hamiltonian = _matrix_one_mode_hamiltonian(0.0, current, params)
    reference_energies, reference_vectors = np.linalg.eigh(reference_hamiltonian)
    retained = reference_vectors[:, :dim]
    reference_ground = float(reference_energies[0])
    identity = np.eye(dim, dtype=float)

    grid = np.linspace(-params.fit_radius, params.fit_radius, params.n_fit_points)
    shell_matrices = np.empty((len(grid), dim, dim), dtype=float)
    for index, point in enumerate(grid):
        hamiltonian = _matrix_one_mode_hamiltonian(point, current, params)
        projected = retained.T @ hamiltonian @ retained
        shell_matrices[index] = params.shell_measure * (
            projected
            - reference_ground * identity
            - matrix_polynomial_pes_value(current, point)
        )
        shell_matrices[index] = 0.5 * (shell_matrices[index] + shell_matrices[index].T)

    basis = np.column_stack(
        [grid**order / factorial(order) for order in range(current.order + 1)]
    )
    targets = shell_matrices.reshape(len(grid), -1)
    coefficients, residuals, _, _ = np.linalg.lstsq(basis, targets, rcond=None)
    fitted = basis @ coefficients
    if residuals.size:
        residual_norm = float(np.sqrt(np.sum(residuals)))
    else:
        residual_norm = float(np.linalg.norm(targets - fitted))
    shell_coefficients = coefficients.reshape(current.order + 1, dim, dim)
    shell_coefficients = 0.5 * (
        shell_coefficients + np.swapaxes(shell_coefficients, 1, 2)
    )
    return MatrixOneModeNARGShellProjection(
        parameters=params,
        state=current,
        grid=grid,
        shell_matrices=shell_matrices,
        shell_coefficients=shell_coefficients,
        residual_norm=residual_norm,
        reference_energies=np.asarray(reference_energies[:dim], dtype=float),
    )


def matrix_one_mode_narg_beta(
    state,
    parameters=None,
    *,
    normalize_gap: bool = True,
    gauge_ground: bool = True,
    **kwargs,
):
    """Normalized matrix analytical-NARG beta function.

    The local running matrix PES is

    ``K_l(phi) = sum_n C_n(l) phi**n / n!``.

    Its coefficient beta function is the local shell part of the full
    nonadiabatic flow,

    ``d_l K = K - d_phi phi d_phi K + A[Pi_shell[K] - K]``.

    The active-coordinate Hamiltonian associated with this matrix PES is built
    separately by :func:`matrix_one_mode_narg_effective_hamiltonian`, where the
    slow kinetic term is overlap-dressed as

    ``T_phi[i,j] <A_i^a|A_j^b>``.

    For ``D > 1`` the overall energy scale is fixed by subtracting
    ``eta K`` with ``eta`` chosen so the first conditional excitation gap in
    ``C_0`` is stationary.  A final scalar gauge shift keeps the ground energy
    derivative zero.
    """
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_matrix_polynomial_state(state)
    d_phi = 0.5 * (1.0 - params.s)
    scaling = np.empty_like(current.coefficients)
    for order, matrix in enumerate(current.coefficients):
        scaling[order] = (1.0 - order * d_phi) * matrix
    projection = matrix_one_mode_narg_shell_projection(current, params)
    beta = scaling + projection.shell_coefficients

    if normalize_gap and current.dimension > 1:
        energies, vectors = np.linalg.eigh(current.coefficients[0])
        gap = float(energies[1] - energies[0])
        if abs(gap) > 1e-12:
            beta0 = vectors.T @ beta[0] @ vectors
            eta = float((beta0[1, 1] - beta0[0, 0]) / gap)
            beta = beta - eta * current.coefficients

    if gauge_ground:
        energies, vectors = np.linalg.eigh(current.coefficients[0])
        beta0 = vectors.T @ beta[0] @ vectors
        beta[0] = beta[0] - float(beta0[0, 0]) * np.eye(current.dimension)

    return 0.5 * (beta + np.swapaxes(beta, 1, 2))


def integrate_matrix_one_mode_narg_flow(
    initial_state,
    parameters=None,
    *,
    lmax: float = 8.0,
    nsteps: int = 100,
    normalize_gap: bool = True,
    gauge_ground: bool = True,
    **kwargs,
):
    """Integrate the matrix-valued one-mode analytical NARG beta function.

    The default flow keeps the lowest constant energy gauged to zero and fixes
    the first conditional excitation gap, matching
    :func:`matrix_one_mode_narg_beta`.
    """
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("nsteps must be positive.")
    lmax = float(lmax)
    if lmax <= 0.0:
        raise ValueError("lmax must be positive.")
    initial = _coerce_matrix_polynomial_state(initial_state).asarray()

    grid = np.linspace(0.0, lmax, nsteps + 1)
    states = np.empty((nsteps + 1,) + initial.shape, dtype=float)
    states[0] = initial
    step = grid[1] - grid[0]
    for index in range(nsteps):
        current = states[index]
        k1 = matrix_one_mode_narg_beta(
            current,
            params,
            normalize_gap=normalize_gap,
            gauge_ground=gauge_ground,
        )
        k2 = matrix_one_mode_narg_beta(
            current + 0.5 * step * k1,
            params,
            normalize_gap=normalize_gap,
            gauge_ground=gauge_ground,
        )
        k3 = matrix_one_mode_narg_beta(
            current + 0.5 * step * k2,
            params,
            normalize_gap=normalize_gap,
            gauge_ground=gauge_ground,
        )
        k4 = matrix_one_mode_narg_beta(
            current + step * k3,
            params,
            normalize_gap=normalize_gap,
            gauge_ground=gauge_ground,
        )
        states[index + 1] = current + (step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        states[index + 1] = 0.5 * (
            states[index + 1] + np.swapaxes(states[index + 1], 1, 2)
        )

    return MatrixOneModeNARGFlow(
        parameters=params,
        l=grid,
        states=states,
        normalize_gap=bool(normalize_gap),
        gauge_ground=bool(gauge_ground),
    )


def _pack_symmetric_matrix_coefficients(coefficients):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 3 or coefficients.shape[1] != coefficients.shape[2]:
        raise ValueError("coefficients must have shape (n, D, D).")
    indices = np.triu_indices(coefficients.shape[1])
    values = [coefficients[order][indices] for order in range(coefficients.shape[0])]
    return np.concatenate(values)


def _unpack_symmetric_matrix_coefficients(vector, *, order: int, dimension: int):
    vector = np.asarray(vector, dtype=float)
    indices = np.triu_indices(dimension)
    block = len(indices[0])
    expected = (int(order) + 1) * block
    if len(vector) != expected:
        raise ValueError(f"expected {expected} packed values, got {len(vector)}.")
    coefficients = np.zeros((int(order) + 1, dimension, dimension), dtype=float)
    for degree in range(int(order) + 1):
        values = vector[degree * block : (degree + 1) * block]
        coefficients[degree][indices] = values
        coefficients[degree] = coefficients[degree] + np.triu(coefficients[degree], 1).T
    return coefficients


def _matrix_polynomial_labels(order: int, dimension: int):
    indices = np.triu_indices(dimension)
    labels = []
    for degree in range(int(order) + 1):
        for row, col in zip(*indices):
            labels.append(f"C{degree}[{row},{col}]")
    return labels


def matrix_one_mode_narg_jacobian(
    state,
    parameters=None,
    *,
    step: float = 1e-5,
    **kwargs,
):
    """Finite-difference Jacobian of the normalized matrix NARG flow."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_matrix_polynomial_state(state)
    center = _pack_symmetric_matrix_coefficients(current.coefficients)
    step = float(step)
    if step <= 0.0:
        raise ValueError("step must be positive.")
    jacobian = np.empty((len(center), len(center)), dtype=float)
    for column in range(len(center)):
        delta = np.zeros_like(center)
        width = step * max(1.0, abs(center[column]))
        delta[column] = width
        plus_state = MatrixPolynomialPESState(
            _unpack_symmetric_matrix_coefficients(
                center + delta,
                order=current.order,
                dimension=current.dimension,
            )
        )
        minus_state = MatrixPolynomialPESState(
            _unpack_symmetric_matrix_coefficients(
                center - delta,
                order=current.order,
                dimension=current.dimension,
            )
        )
        plus = _pack_symmetric_matrix_coefficients(
            matrix_one_mode_narg_beta(plus_state, params)
        )
        minus = _pack_symmetric_matrix_coefficients(
            matrix_one_mode_narg_beta(minus_state, params)
        )
        jacobian[:, column] = (plus - minus) / (2.0 * width)
    return jacobian


def matrix_one_mode_narg_linearization(
    state,
    parameters=None,
    *,
    step: float = 1e-5,
    **kwargs,
):
    """Linearize the normalized matrix one-mode NARG flow."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    fixed_point = _coerce_matrix_polynomial_state(state)
    jacobian = matrix_one_mode_narg_jacobian(fixed_point, params, step=step)
    eigenvalues = np.linalg.eigvals(jacobian)
    if np.max(np.abs(np.imag(eigenvalues))) <= 1e-10:
        eigenvalues = np.real(eigenvalues)
    return MatrixOneModeNARGLinearization(
        parameters=params,
        fixed_point=fixed_point,
        jacobian=jacobian,
        eigenvalues=np.asarray(eigenvalues),
        packed_labels=_matrix_polynomial_labels(fixed_point.order, fixed_point.dimension),
    )


def _normalize_matrix_coefficients(coefficients, *, normalize_gap: bool = True):
    coefficients = np.asarray(coefficients, dtype=float)
    coefficients = 0.5 * (coefficients + np.swapaxes(coefficients, 1, 2))
    energies, vectors = np.linalg.eigh(coefficients[0])
    coefficients[0] = coefficients[0] - float(energies[0]) * np.eye(coefficients.shape[1])
    gap = np.nan
    if normalize_gap and coefficients.shape[1] > 1:
        shifted = vectors.T @ coefficients[0] @ vectors
        gap = float(shifted[1, 1] - shifted[0, 0])
        if abs(gap) > 1e-14:
            coefficients = coefficients / gap
    return coefficients, gap


def _discrete_matrix_wilson_step(
    state,
    *,
    site: int,
    onsite: float,
    coupling: float,
    block_scale: float,
    shell_scale: float,
    n_conditional_states: int,
    polynomial_order: int,
    coordinate_basis_size: int,
    fit_radius: float,
    n_fit_points: int,
    normalize_gap: bool,
):
    current = _coerce_matrix_polynomial_state(state)
    old_dim = current.dimension
    coordinate, kinetic = _fock_coordinate_and_kinetic(coordinate_basis_size)
    identity_old = np.eye(old_dim, dtype=float)
    old_h0 = np.kron(kinetic, identity_old) + _matrix_polynomial_pes_operator(
        current,
        coordinate,
    )
    old_h0 = 0.5 * (old_h0 + old_h0.T)
    retained_energies, retained_vectors = np.linalg.eigh(old_h0)
    keep = min(int(n_conditional_states), len(retained_energies))
    retained = retained_vectors[:, :keep]
    reference_ground = float(retained_energies[0])
    identity_keep = np.eye(keep, dtype=float)
    boundary_operator = np.kron(coordinate, identity_old)

    grid = np.linspace(-float(fit_radius), float(fit_radius), int(n_fit_points))
    force_factor = (float(shell_scale) / float(block_scale)) * float(coupling)
    matrices = np.empty((len(grid), keep, keep), dtype=float)
    for index, point in enumerate(grid):
        forced = old_h0 + force_factor * float(point) * boundary_operator
        projected = retained.T @ forced @ retained
        matrices[index] = (
            float(block_scale) * (projected - reference_ground * identity_keep)
            + 0.5 * float(shell_scale) * float(onsite) * float(point) ** 2 * identity_keep
        )
        matrices[index] = 0.5 * (matrices[index] + matrices[index].T)

    basis = np.column_stack(
        [grid**order / factorial(order) for order in range(int(polynomial_order) + 1)]
    )
    targets = matrices.reshape(len(grid), -1)
    coefficients, residuals, _, _ = np.linalg.lstsq(basis, targets, rcond=None)
    fitted = basis @ coefficients
    if residuals.size:
        residual_norm = float(np.sqrt(np.sum(residuals)))
    else:
        residual_norm = float(np.linalg.norm(targets - fitted))
    next_coefficients = coefficients.reshape(int(polynomial_order) + 1, keep, keep)
    next_coefficients = 0.5 * (
        next_coefficients + np.swapaxes(next_coefficients, 1, 2)
    )
    next_coefficients, gap = _normalize_matrix_coefficients(
        next_coefficients,
        normalize_gap=normalize_gap,
    )
    return DiscreteMatrixWilsonNARGStep(
        site=int(site),
        onsite=float(onsite),
        coupling=float(coupling),
        block_scale=float(block_scale),
        shell_scale=float(shell_scale),
        coefficients=next_coefficients,
        residual_norm=residual_norm,
        retained_energies=np.asarray(retained_energies[:keep], dtype=float),
        normalization_gap=float(gap),
    )


def discrete_matrix_wilson_narg_flow(
    chain,
    initial_state,
    *,
    n_conditional_states: int = 2,
    polynomial_order: int = 4,
    coordinate_basis_size: int = 14,
    fit_radius: float = 0.75,
    n_fit_points: int = 19,
    initial_coupling: float | None = None,
    nrg_rescale: bool = True,
    Lambda: float | None = None,
    rescale_power: float = 1.0,
    normalize_gap: bool = True,
):
    """Run a coordinate-coupled Wilson-chain matrix PES closure.

    The running object is a matrix Taylor PES for the current Wilson boundary
    coordinate.  To add the next Wilson site, the previous boundary coordinate
    is represented in a small oscillator basis, coupled to the new coordinate,
    projected onto ``n_conditional_states`` retained conditional states, and
    refit as a matrix Taylor PES of the new coordinate.

    This is not the exact Wilson-chain NARG update for a bosonic chain.  The
    exact update is implemented by :class:`SpinBosonWilsonNARG` in
    ``spin_boson.py`` and uses the projected boundary annihilation operator:

    ``H' = H_block ⊗ I + eps I ⊗ b^dagger b
           + t(B^dagger ⊗ b + B ⊗ b^dagger)``.
    """
    current = _coerce_matrix_polynomial_state(initial_state)
    n_conditional_states = int(n_conditional_states)
    polynomial_order = int(polynomial_order)
    coordinate_basis_size = int(coordinate_basis_size)
    n_fit_points = int(n_fit_points)
    if n_conditional_states < 1:
        raise ValueError("n_conditional_states must be positive.")
    if polynomial_order < 2:
        raise ValueError("polynomial_order must be at least two.")
    if coordinate_basis_size < 2:
        raise ValueError("coordinate_basis_size must be at least two.")
    if n_fit_points < polynomial_order + 1:
        raise ValueError("n_fit_points must be larger than the polynomial order.")
    if nrg_rescale:
        if Lambda is None:
            raise ValueError("Lambda is required when nrg_rescale=True.")
        if float(Lambda) <= 1.0:
            raise ValueError("Lambda must be larger than one.")

    coupling0 = (
        _default_initial_coupling(chain)
        if initial_coupling is None
        else float(initial_coupling)
    )
    steps = []
    for site, onsite in enumerate(np.asarray(chain.onsite, dtype=float)):
        shell_scale = _scale_for_site(
            site,
            nrg_rescale=nrg_rescale,
            Lambda=Lambda,
            rescale_power=rescale_power,
        )
        if site == 0:
            block_scale = 1.0
            coupling = coupling0
        else:
            previous_scale = _scale_for_site(
                site - 1,
                nrg_rescale=nrg_rescale,
                Lambda=Lambda,
                rescale_power=rescale_power,
            )
            block_scale = shell_scale / previous_scale
            coupling = float(chain.hopping[site - 1])
        step = _discrete_matrix_wilson_step(
            current,
            site=site,
            onsite=float(onsite),
            coupling=float(coupling),
            block_scale=float(block_scale),
            shell_scale=float(shell_scale),
            n_conditional_states=n_conditional_states,
            polynomial_order=polynomial_order,
            coordinate_basis_size=coordinate_basis_size,
            fit_radius=fit_radius,
            n_fit_points=n_fit_points,
            normalize_gap=normalize_gap,
        )
        steps.append(step)
        current = MatrixPolynomialPESState(step.coefficients)

    return DiscreteMatrixWilsonNARGFlow(
        initial_state=_coerce_matrix_polynomial_state(initial_state),
        steps=steps,
        n_conditional_states=n_conditional_states,
        polynomial_order=polynomial_order,
        nrg_rescaled=bool(nrg_rescale),
        Lambda=None if Lambda is None else float(Lambda),
        rescale_power=float(rescale_power),
    )


def polynomial_one_mode_narg_shell_projection(state, parameters=None, **kwargs):
    """Project the one-mode NARG shell onto the state's Taylor order."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_polynomial_state(state)
    grid = np.linspace(-params.fit_radius, params.fit_radius, params.n_fit_points)
    shell = polynomial_one_mode_narg_shell_energy(grid, current, params)
    basis = np.column_stack(
        [grid**index / factorial(index) for index in range(1, current.order + 1)]
    )
    shell_couplings, residuals, _, _ = np.linalg.lstsq(basis, shell, rcond=None)
    fitted = basis @ shell_couplings
    if residuals.size:
        residual_norm = float(np.sqrt(residuals[0]))
    else:
        residual_norm = float(np.linalg.norm(shell - fitted))
    return PolynomialNARGShellProjection(
        parameters=params,
        state=current,
        grid=grid,
        shell_energy=shell,
        shell_couplings=np.asarray(shell_couplings, dtype=float),
        residual_norm=residual_norm,
    )


def polynomial_one_mode_narg_beta(state, parameters=None, **kwargs):
    """Higher-order NARG beta function from one-mode eigenvalue projection."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    current = _coerce_polynomial_state(state)
    d_phi = 0.5 * (1.0 - params.s)
    scaling = np.array(
        [
            (1.0 - (index + 1) * d_phi) * coupling
            for index, coupling in enumerate(current.couplings)
        ],
        dtype=float,
    )
    projection = polynomial_one_mode_narg_shell_projection(current, params)
    return scaling + projection.shell_couplings


def polynomial_one_mode_narg_jacobian(
    state,
    parameters=None,
    *,
    step: float = 1e-5,
    **kwargs,
):
    """Finite-difference Jacobian of the higher-order NARG beta function."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    center = _coerce_polynomial_state(state).asarray()
    step = float(step)
    if step <= 0.0:
        raise ValueError("step must be positive.")
    jacobian = np.empty((len(center), len(center)), dtype=float)
    for column in range(len(center)):
        delta = np.zeros_like(center)
        width = step * max(1.0, abs(center[column]))
        delta[column] = width
        plus = polynomial_one_mode_narg_beta(center + delta, params)
        minus = polynomial_one_mode_narg_beta(center - delta, params)
        jacobian[:, column] = (plus - minus) / (2.0 * width)
    return jacobian


def polynomial_one_mode_narg_linearization(
    state,
    parameters=None,
    *,
    step: float = 1e-5,
    **kwargs,
):
    """Linearize the higher-order NARG flow around a supplied fixed point."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    fixed_point = _coerce_polynomial_state(state)
    jacobian = polynomial_one_mode_narg_jacobian(fixed_point, params, step=step)
    eigenvalues = np.linalg.eigvals(jacobian)
    if np.max(np.abs(np.imag(eigenvalues))) <= 1e-10:
        eigenvalues = np.real(eigenvalues)
    return PolynomialNARGLinearization(
        parameters=params,
        fixed_point=fixed_point,
        jacobian=jacobian,
        eigenvalues=np.asarray(eigenvalues),
    )


def integrate_polynomial_one_mode_narg_flow(
    initial_state,
    parameters=None,
    *,
    lmax: float = 8.0,
    nsteps: int = 100,
    **kwargs,
):
    """Integrate the higher-order one-mode NARG beta functions with RK4."""
    params = _coerce_one_mode_parameters(parameters, **kwargs)
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("nsteps must be positive.")
    lmax = float(lmax)
    if lmax <= 0.0:
        raise ValueError("lmax must be positive.")
    state = _coerce_polynomial_state(initial_state).asarray()

    grid = np.linspace(0.0, lmax, nsteps + 1)
    states = np.empty((nsteps + 1, len(state)), dtype=float)
    states[0] = state
    step = grid[1] - grid[0]
    for index in range(nsteps):
        current = states[index]
        k1 = polynomial_one_mode_narg_beta(current, params)
        k2 = polynomial_one_mode_narg_beta(current + 0.5 * step * k1, params)
        k3 = polynomial_one_mode_narg_beta(current + 0.5 * step * k2, params)
        k4 = polynomial_one_mode_narg_beta(current + step * k3, params)
        states[index + 1] = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return PolynomialNARGFlow(parameters=params, l=grid, states=states)


def landau_critical_exponents(s: float):
    """Return closed analytical NARG/Landau critical exponents.

    ``y_t``, ``y_u`` and ``y_h`` are the Gaussian RG eigenvalues.  The
    ``beta``, ``gamma`` and ``delta`` fields are the Landau equation-of-state
    exponents.  The hyperscaling values are included separately because the
    quartic coupling is dangerously irrelevant for ``s < 1/2``.
    """
    s = float(s)
    if s <= 0.0:
        raise ValueError("s must be positive.")
    y_t = s
    y_u = 2.0 * s - 1.0
    y_h = 0.5 * (1.0 + s)
    nu = 1.0 / y_t
    hyperscaling_beta = (1.0 - y_h) / y_t
    hyperscaling_gamma = (2.0 * y_h - 1.0) / y_t
    hyperscaling_delta = y_h / (1.0 - y_h)
    return AnalyticalLandauCriticalExponents(
        s=s,
        y_t=y_t,
        y_u=y_u,
        y_h=y_h,
        nu=nu,
        beta=0.5,
        gamma=1.0,
        delta=3.0,
        hyperscaling_beta=hyperscaling_beta,
        hyperscaling_gamma=hyperscaling_gamma,
        hyperscaling_delta=hyperscaling_delta,
    )
