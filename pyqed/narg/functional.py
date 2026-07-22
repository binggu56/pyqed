"""Schrodinger-wavefunctional NARG toys.

This module contains a small continuum toy for testing the idea of NARG as a
direct wavefunction compression.  The bosonic coordinate is continuous, the
fermionic state is a conditional one-particle Gaussian/Slater state, and the
NARG branches are obtained from a Schmidt decomposition of the wavefunction
kernel rather than from an effective action.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])


def hermite_quadrature(order: int, omega: float = 1.0):
    """Gauss-Hermite quadrature nodes and weights for plain ``dq`` integrals."""
    order = int(order)
    omega = float(omega)
    if order < 1:
        raise ValueError("order must be positive.")
    if omega <= 0:
        raise ValueError("omega must be positive.")
    x, weights = hermgauss(order)
    q = x / np.sqrt(omega)
    plain_weights = weights * np.exp(x * x) / np.sqrt(omega)
    return q, plain_weights


def hermite_function_values(q, nmax: int, omega: float = 1.0):
    """Normalized harmonic-oscillator basis functions at ``q``."""
    q = np.asarray(q, dtype=float)
    nmax = int(nmax)
    omega = float(omega)
    if nmax < 1:
        raise ValueError("nmax must be positive.")
    if omega <= 0:
        raise ValueError("omega must be positive.")

    x = np.sqrt(omega) * q
    values = np.zeros((q.size, nmax), dtype=float)
    values[:, 0] = (omega / np.pi) ** 0.25 * np.exp(-0.5 * x * x)
    if nmax == 1:
        return values
    values[:, 1] = np.sqrt(2.0) * x * values[:, 0]
    for n in range(1, nmax - 1):
        values[:, n + 1] = (
            np.sqrt(2.0 / (n + 1.0)) * x * values[:, n]
            - np.sqrt(n / (n + 1.0)) * values[:, n - 1]
        )
    return values


def oscillator_operators(nbasis: int, omega: float = 1.0):
    """Return ``q``, ``p``, and oscillator Hamiltonian matrices."""
    nbasis = int(nbasis)
    omega = float(omega)
    if nbasis < 1:
        raise ValueError("nbasis must be positive.")
    if omega <= 0:
        raise ValueError("omega must be positive.")

    destroy = np.zeros((nbasis, nbasis), dtype=complex)
    for n in range(1, nbasis):
        destroy[n - 1, n] = np.sqrt(n)
    create = destroy.T.conj()
    q = (destroy + create) / np.sqrt(2.0 * omega)
    p = 1j * np.sqrt(omega / 2.0) * (create - destroy)
    h = 0.5 * (p @ p) + 0.5 * omega * omega * (q @ q)
    return q, p, 0.5 * (h + h.T.conj())


@dataclass
class ConditionalGaussianNARGResult:
    """Result of one continuous conditional-Gaussian NARG compression."""

    rank: int
    q: np.ndarray
    weights: np.ndarray
    singular_values: np.ndarray
    boson_branches: np.ndarray
    fermion_branches: np.ndarray
    wavefunction_values: np.ndarray
    oscillator_coefficients: np.ndarray
    energy: float
    exact_energy: float
    discarded_weight: float
    coefficient_norm: float

    @property
    def kept_weight(self) -> float:
        return float(np.sum(self.singular_values[: self.rank] ** 2))


class ConditionalGaussianWavefunctionNARG:
    """Continuum toy for direct wavefunction NARG.

    The Hamiltonian is a one-boson-coordinate, one-fermion two-level toy,

    ``H = H_b + epsilon sigma_z + (mixing + coupling q) sigma_x``.

    For every continuous ``q``, the fermion Gaussian is the occupied lowest
    one-particle spinor of the conditional two-level Hamiltonian.  The bosonic
    reference is a harmonic-oscillator Gaussian.  The NARG step is the Schmidt
    compression of ``chi(q) |Omega_F(q)>`` over the continuous ``q`` measure.
    """

    def __init__(
        self,
        *,
        oscillator_frequency: float = 1.0,
        fermion_gap: float = 0.45,
        coupling: float = 1.1,
        mixing: float = 0.2,
        nbasis: int = 36,
        quadrature_order: int = 120,
    ):
        self.oscillator_frequency = float(oscillator_frequency)
        self.fermion_gap = float(fermion_gap)
        self.coupling = float(coupling)
        self.mixing = float(mixing)
        self.nbasis = int(nbasis)
        self.quadrature_order = int(quadrature_order)
        if self.oscillator_frequency <= 0:
            raise ValueError("oscillator_frequency must be positive.")
        if self.nbasis < 1:
            raise ValueError("nbasis must be positive.")
        if self.quadrature_order < 1:
            raise ValueError("quadrature_order must be positive.")

    def quadrature(self):
        return hermite_quadrature(self.quadrature_order, self.oscillator_frequency)

    def boson_reference(self, q):
        q = np.asarray(q, dtype=float)
        omega = self.oscillator_frequency
        return (omega / np.pi) ** 0.25 * np.exp(-0.5 * omega * q * q)

    def fermion_hamiltonian(self, q):
        return self.fermion_gap * SIGMA_Z + (self.mixing + self.coupling * float(q)) * SIGMA_X

    def conditional_fermion_gaussians(self, q):
        """Lowest occupied one-particle spinors, phase-aligned along ``q``."""
        q = np.asarray(q, dtype=float)
        spinors = np.empty((q.size, 2), dtype=float)
        previous = None
        for index, value in enumerate(q):
            _, vectors = np.linalg.eigh(self.fermion_hamiltonian(value))
            spinor = np.real(vectors[:, 0])
            if previous is None:
                if spinor[0] < 0:
                    spinor = -spinor
            elif np.dot(previous, spinor) < 0:
                spinor = -spinor
            spinors[index] = spinor
            previous = spinor
        return spinors

    def conditional_wavefunction(self, normalize: bool = True):
        """Return quadrature values of ``chi(q)|Omega_F(q)>``."""
        q, weights = self.quadrature()
        values = self.boson_reference(q)[:, None] * self.conditional_fermion_gaussians(q)
        if normalize:
            norm = np.sqrt(np.sum(weights * np.sum(np.abs(values) ** 2, axis=1)))
            if norm == 0:
                raise ValueError("conditional wavefunction has zero norm.")
            values = values / norm
        return q, weights, values

    def schmidt_compress(self, rank: int):
        """Schmidt-compress the continuous conditional manifold to ``rank``."""
        rank = int(rank)
        if rank < 1:
            raise ValueError("rank must be positive.")

        q, weights, values = self.conditional_wavefunction(normalize=True)
        weighted = np.sqrt(weights)[:, None] * values
        left, singular_values, right_h = np.linalg.svd(weighted, full_matrices=False)
        rank = min(rank, singular_values.size)

        weighted_compressed = (left[:, :rank] * singular_values[:rank]) @ right_h[:rank]
        compressed_values = weighted_compressed / np.sqrt(weights)[:, None]
        branches = left[:, :rank] / np.sqrt(weights)[:, None]
        oscillator_coefficients = self.oscillator_coefficients(compressed_values, q, weights)
        energy, coeff_norm = self.energy_from_coefficients(oscillator_coefficients)
        exact = self.exact_ground_energy()
        discarded = float(np.sum(singular_values[rank:] ** 2))

        return ConditionalGaussianNARGResult(
            rank=rank,
            q=q,
            weights=weights,
            singular_values=singular_values,
            boson_branches=branches,
            fermion_branches=right_h[:rank].copy(),
            wavefunction_values=compressed_values,
            oscillator_coefficients=oscillator_coefficients,
            energy=float(energy),
            exact_energy=float(exact),
            discarded_weight=discarded,
            coefficient_norm=float(coeff_norm),
        )

    def oscillator_coefficients(self, wavefunction_values, q=None, weights=None):
        """Project two-component continuum values onto oscillator basis."""
        if q is None or weights is None:
            q, weights = self.quadrature()
        basis = hermite_function_values(q, self.nbasis, self.oscillator_frequency)
        return basis.T @ (weights[:, None] * np.asarray(wavefunction_values))

    def hamiltonian_matrix(self):
        q, _, h_boson = oscillator_operators(self.nbasis, self.oscillator_frequency)
        h_fermion = self.fermion_gap * SIGMA_Z + self.mixing * SIGMA_X
        return (
            np.kron(h_boson, np.eye(2))
            + np.kron(np.eye(self.nbasis), h_fermion)
            + self.coupling * np.kron(q, SIGMA_X)
        )

    def energy_from_coefficients(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=complex)
        vector = coefficients.reshape(-1)
        norm = np.vdot(vector, vector)
        if abs(norm) == 0:
            raise ValueError("coefficient vector has zero norm.")
        hamiltonian = self.hamiltonian_matrix()
        energy = np.vdot(vector, hamiltonian @ vector) / norm
        return float(np.real(energy)), float(np.real(norm))

    def exact_ground_energy(self):
        return float(np.linalg.eigvalsh(self.hamiltonian_matrix())[0])


def interval_legendre_quadrature(order: int, length: float):
    """Gauss-Legendre quadrature on ``[0, length]``."""
    order = int(order)
    length = float(length)
    if order < 1:
        raise ValueError("order must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")
    nodes, weights = leggauss(order)
    x = 0.5 * length * (nodes + 1.0)
    w = 0.5 * length * weights
    return x, w


def sine_basis_values(x, nmodes: int, length: float):
    """Dirichlet sine basis values on a continuum interval."""
    x = np.asarray(x, dtype=float)
    nmodes = int(nmodes)
    length = float(length)
    modes = np.arange(1, nmodes + 1, dtype=float)
    return np.sqrt(2.0 / length) * np.sin(np.pi * x[:, None] * modes[None, :] / length)


def sine_basis_derivative_values(x, nmodes: int, length: float):
    """Spatial derivatives of the Dirichlet sine basis."""
    x = np.asarray(x, dtype=float)
    nmodes = int(nmodes)
    length = float(length)
    modes = np.arange(1, nmodes + 1, dtype=float)
    wave_numbers = np.pi * modes / length
    return (
        np.sqrt(2.0 / length)
        * wave_numbers[None, :]
        * np.cos(x[:, None] * wave_numbers[None, :])
    )


def sine_basis_derivative_matrix(nmodes: int, length: float):
    """Projected first-derivative matrix in the Dirichlet sine basis."""
    nmodes = int(nmodes)
    length = float(length)
    if nmodes < 1:
        raise ValueError("nmodes must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")
    modes = np.arange(1, nmodes + 1, dtype=float)
    left, right = np.meshgrid(modes, modes, indexing="ij")
    parity = (left + right) % 2
    matrix = np.zeros((nmodes, nmodes), dtype=float)
    mask = parity == 1
    matrix[mask] = 4.0 * left[mask] * right[mask] / (length * (left[mask] ** 2 - right[mask] ** 2))
    return matrix


def sine_dvr_grid(npoints: int, length: float):
    """Sine-DVR grid and uniform quadrature weights on ``[0, length]``."""
    npoints = int(npoints)
    length = float(length)
    if npoints < 1:
        raise ValueError("npoints must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")
    points = np.arange(1, npoints + 1, dtype=float)
    x = length * points / (npoints + 1.0)
    weights = np.full(npoints, length / (npoints + 1.0), dtype=float)
    return x, weights


def sine_dvr_transform(npoints: int):
    """Orthogonal transform from sine basis modes to sine-DVR sites."""
    npoints = int(npoints)
    if npoints < 1:
        raise ValueError("npoints must be positive.")
    sites = np.arange(1, npoints + 1, dtype=float)[:, None]
    modes = np.arange(1, npoints + 1, dtype=float)[None, :]
    return np.sqrt(2.0 / (npoints + 1.0)) * np.sin(np.pi * sites * modes / (npoints + 1.0))


def sine_dvr_kinetic_matrix(npoints: int, length: float, mass: float = 1.0):
    """Single-particle sine-DVR KEO ``-1/(2m) d^2/dx^2``."""
    npoints = int(npoints)
    length = float(length)
    mass = float(mass)
    if mass <= 0:
        raise ValueError("mass must be positive.")
    transform = sine_dvr_transform(npoints)
    modes = np.arange(1, npoints + 1, dtype=float)
    eigenvalues = 0.5 * (np.pi * modes / length) ** 2 / mass
    kinetic = transform @ np.diag(eigenvalues) @ transform.T
    return 0.5 * (kinetic + kinetic.T)


def sine_dvr_derivative_matrix(npoints: int, length: float):
    """Sine-DVR first derivative matrix for the Dirac-like kinetic term."""
    transform = sine_dvr_transform(npoints)
    derivative = transform @ sine_basis_derivative_matrix(npoints, length) @ transform.T
    return 0.5 * (derivative - derivative.T)


def periodic_sinc_grid(npoints: int, length: float):
    """Periodic sinc-DVR spatial grid and uniform quadrature weights."""
    npoints = int(npoints)
    length = float(length)
    if npoints < 1:
        raise ValueError("npoints must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")
    x = length * np.arange(npoints, dtype=float) / npoints
    weights = np.full(npoints, length / npoints, dtype=float)
    return x, weights


def periodic_real_fourier_transform(npoints: int, length: float):
    """Real orthogonal Fourier transform for periodic sinc-DVR fields.

    Columns are ordered as ``0, cos(1), sin(1), cos(2), sin(2), ...`` with a
    final Nyquist cosine for even ``npoints``.  This keeps each degenerate
    ``+/-k`` pair adjacent as real cosine/sine coordinates.
    """
    npoints = int(npoints)
    length = float(length)
    if npoints < 1:
        raise ValueError("npoints must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")

    sites = np.arange(npoints, dtype=float)
    columns = [np.ones(npoints, dtype=float) / np.sqrt(npoints)]
    wave_numbers = [0.0]
    labels = [("zero", 0)]
    max_pair = (npoints - 1) // 2
    for mode in range(1, max_pair + 1):
        phase = 2.0 * np.pi * mode * sites / npoints
        columns.append(np.sqrt(2.0 / npoints) * np.cos(phase))
        wave_numbers.append(2.0 * np.pi * mode / length)
        labels.append(("cos", mode))
        columns.append(np.sqrt(2.0 / npoints) * np.sin(phase))
        wave_numbers.append(2.0 * np.pi * mode / length)
        labels.append(("sin", mode))
    if npoints % 2 == 0:
        columns.append((-1.0) ** sites / np.sqrt(npoints))
        wave_numbers.append(np.pi * npoints / length)
        labels.append(("nyquist", npoints // 2))
    transform = np.column_stack(columns)
    return transform, np.asarray(wave_numbers, dtype=float), labels


def fixed_particle_basis(norbitals: int, nparticles: int):
    """Occupation tuples for a fixed-particle fermion sector."""
    norbitals = int(norbitals)
    nparticles = int(nparticles)
    if norbitals < 1:
        raise ValueError("norbitals must be positive.")
    if nparticles < 0 or nparticles > norbitals:
        raise ValueError("nparticles must lie between 0 and norbitals.")
    return list(combinations(range(norbitals), nparticles))


def one_body_sector_matrix(one_body, nparticles: int):
    """Second-quantized one-body operator in a fixed-particle sector."""
    one_body = np.asarray(one_body, dtype=complex)
    norbitals = one_body.shape[0]
    if one_body.shape != (norbitals, norbitals):
        raise ValueError("one_body must be square.")
    basis = fixed_particle_basis(norbitals, nparticles)
    index = {state: idx for idx, state in enumerate(basis)}
    out = np.zeros((len(basis), len(basis)), dtype=complex)
    for ket_id, ket in enumerate(basis):
        occupied = list(ket)
        for q in range(norbitals):
            if q not in occupied:
                continue
            q_position = occupied.index(q)
            remove_sign = -1 if (q_position % 2) else 1
            reduced = occupied[:q_position] + occupied[q_position + 1 :]
            for p in range(norbitals):
                coeff = one_body[p, q]
                if coeff == 0 or p in reduced:
                    continue
                insert_position = sum(orb < p for orb in reduced)
                create_sign = -1 if (insert_position % 2) else 1
                bra = tuple(reduced[:insert_position] + [p] + reduced[insert_position:])
                out[index[bra], ket_id] += coeff * remove_sign * create_sign
    return out


def slater_sector_vector(occupied_orbitals, basis):
    """Fock-sector vector for a Slater determinant."""
    occupied_orbitals = np.asarray(occupied_orbitals, dtype=complex)
    coeff = np.empty(len(basis), dtype=complex)
    for idx, state in enumerate(basis):
        coeff[idx] = _small_determinant(occupied_orbitals[list(state), :])
    norm = np.linalg.norm(coeff)
    if norm == 0:
        raise ValueError("Slater determinant has zero norm in this sector.")
    return coeff / norm


def _small_determinant(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    n = matrix.shape[0]
    if n == 0:
        return 1.0 + 0.0j
    if n == 1:
        return matrix[0, 0]
    if n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    det = 0.0 + 0.0j
    for perm in permutations(range(n)):
        inversions = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
        sign = -1 if inversions % 2 else 1
        term = 1.0 + 0.0j
        for row, col in enumerate(perm):
            term *= matrix[row, col]
        det += sign * term
    return det


def product_oscillator_basis_values(q_samples, nbasis: int, frequencies):
    """Product harmonic-oscillator basis functions for scalar mode samples."""
    q_samples = np.asarray(q_samples, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if q_samples.ndim != 2 or q_samples.shape[1] != frequencies.size:
        raise ValueError("q_samples must have shape (nsamples, nmodes).")
    basis_by_mode = [
        hermite_function_values(q_samples[:, mode], nbasis, frequencies[mode])
        for mode in range(frequencies.size)
    ]
    states = list(product(range(int(nbasis)), repeat=frequencies.size))
    values = np.ones((q_samples.shape[0], len(states)), dtype=float)
    for col, state in enumerate(states):
        for mode, quantum in enumerate(state):
            values[:, col] *= basis_by_mode[mode][:, quantum]
    return values, states


def _kron_all(operators):
    out = np.asarray(operators[0])
    for op in operators[1:]:
        out = np.kron(out, np.asarray(op))
    return out


@dataclass
class Yukawa1DNARGResult:
    """Result for the 1+1D direct wavefunctional NARG toy."""

    rank: int
    field_coordinates: np.ndarray
    weights: np.ndarray
    singular_values: np.ndarray
    boson_branches: np.ndarray
    fermion_branches: np.ndarray
    wavefunction_values: np.ndarray
    oscillator_coefficients: np.ndarray
    energy: float
    exact_energy: float
    discarded_weight: float
    coefficient_norm: float
    fermion_basis: list
    boson_basis: list

    @property
    def kept_weight(self) -> float:
        return float(np.sum(self.singular_values[: self.rank] ** 2))


@dataclass
class Yukawa1DVariationalRank1Result:
    """Result of optimizing one conditional-Gaussian wavefunctional branch."""

    widths: np.ndarray
    centers: np.ndarray
    energy: float
    exact_energy: float
    coefficient_norm: float
    oscillator_coefficients: np.ndarray
    success: bool
    message: str
    nfev: int


@dataclass
class Yukawa1DGaussianResponseResult:
    """Analytic Gaussian-response energy for one conditional branch."""

    widths: np.ndarray
    centers: np.ndarray
    energy: float
    exact_energy: float
    boson_energy: float
    fermion_energy: float
    fluctuation_energy: float
    born_huang_energy: float
    fermion_gradient: np.ndarray
    fermion_hessian: np.ndarray
    quantum_metric: np.ndarray
    metric_source: str = "overlap"
    success: bool = True
    message: str = ""
    nfev: int = 1

    @property
    def overlap_metric_energy(self) -> float:
        return self.born_huang_energy


@dataclass
class Yukawa1DGaussianPacketResult:
    """Generalized-eigenvalue result in a nonorthogonal Gaussian packet basis."""

    widths: np.ndarray
    centers: np.ndarray
    energy: float
    exact_energy: float
    coefficients: np.ndarray
    hamiltonian: np.ndarray
    overlap: np.ndarray
    parts: dict


@dataclass
class Yukawa1DRegulatedKineticRank1Result:
    """Rank-1 chi energy using a heat-kernel regulated ``T S`` product."""

    widths: np.ndarray
    centers: np.ndarray
    cutoff: float
    shift: float
    kinetic_weights: np.ndarray
    energy: float
    exact_energy: float
    kinetic_energy: float
    boson_potential_energy: float
    fermion_energy: float
    norm: float

    @property
    def potential_energy(self) -> float:
        return self.boson_potential_energy + self.fermion_energy


@dataclass
class Phi4NARGEffectiveHamiltonianResult:
    """NARG effective Hamiltonian for a two-site ``phi^4`` lattice toy."""

    hamiltonian: np.ndarray
    active_grid: np.ndarray
    environment_grid: np.ndarray
    active_kinetic: np.ndarray
    conditional_states: np.ndarray
    conditional_blocks: np.ndarray
    kinetic_dressing: np.ndarray
    effective_energies: np.ndarray
    exact_energies: np.ndarray
    nbranches: int


class Phi4TwoSiteNARG:
    """Two-site lattice ``phi^4`` model with a conditional NARG ``H_eff``.

    One field coordinate is retained as the active coordinate ``q`` and the
    second is treated as the conditional environment coordinate ``y``.
    """

    def __init__(
        self,
        *,
        active_npoints: int = 11,
        environment_npoints: int = 13,
        field_range: float = 5.0,
        mass2: float = 0.5,
        coupling: float = 0.5,
        stiffness: float = 0.4,
    ):
        self.active_npoints = int(active_npoints)
        self.environment_npoints = int(environment_npoints)
        self.field_range = float(field_range)
        self.mass2 = float(mass2)
        self.coupling = float(coupling)
        self.stiffness = float(stiffness)
        if self.active_npoints < 1 or self.environment_npoints < 1:
            raise ValueError("DVR point counts must be positive.")
        if self.field_range <= 0:
            raise ValueError("field_range must be positive.")
        if self.coupling < 0:
            raise ValueError("coupling must be nonnegative.")
        if self.stiffness < 0:
            raise ValueError("stiffness must be nonnegative.")

        length = 2.0 * self.field_range
        active_grid, _ = sine_dvr_grid(self.active_npoints, length)
        environment_grid, _ = sine_dvr_grid(self.environment_npoints, length)
        self.active_grid = active_grid - self.field_range
        self.environment_grid = environment_grid - self.field_range
        self.active_kinetic = sine_dvr_kinetic_matrix(self.active_npoints, length)
        self.environment_kinetic = sine_dvr_kinetic_matrix(self.environment_npoints, length)
        self._exact_energies_cache = None

    def potential(self, q, y):
        q = np.asarray(q, dtype=float)
        y = np.asarray(y, dtype=float)
        return (
            0.5 * self.mass2 * (q * q + y * y)
            + 0.5 * self.stiffness * (q - y) * (q - y)
            + self.coupling * (q**4 + y**4) / 24.0
        )

    def conditional_environment_hamiltonian(self, q_value):
        potential = self.potential(float(q_value), self.environment_grid)
        return self.environment_kinetic + np.diag(potential)

    def full_hamiltonian_matrix(self):
        identity_active = np.eye(self.active_npoints)
        identity_environment = np.eye(self.environment_npoints)
        q = self.active_grid[:, None]
        y = self.environment_grid[None, :]
        potential = self.potential(q, y).reshape(-1)
        hamiltonian = (
            np.kron(self.active_kinetic, identity_environment)
            + np.kron(identity_active, self.environment_kinetic)
            + np.diag(potential)
        )
        return 0.5 * (hamiltonian + hamiltonian.T)

    def exact_energies(self, nroots: int | None = None):
        if self._exact_energies_cache is None:
            self._exact_energies_cache = np.linalg.eigvalsh(self.full_hamiltonian_matrix())
        if nroots is None:
            return self._exact_energies_cache.copy()
        return self._exact_energies_cache[: int(nroots)].copy()

    @staticmethod
    def _align_subspace(previous, current):
        overlap = previous.T @ current
        left, _, right_h = np.linalg.svd(overlap)
        rotation = right_h.T @ left.T
        return current @ rotation

    def conditional_environment_states(self, nbranches: int = 1):
        nbranches = int(nbranches)
        if nbranches < 1 or nbranches > self.environment_npoints:
            raise ValueError("nbranches must be between 1 and environment_npoints.")
        states = np.empty((self.active_npoints, self.environment_npoints, nbranches), dtype=float)
        blocks = np.empty((self.active_npoints, nbranches, nbranches), dtype=float)
        previous = None
        for index, q_value in enumerate(self.active_grid):
            h_env = self.conditional_environment_hamiltonian(q_value)
            _, vectors = np.linalg.eigh(h_env)
            current = vectors[:, :nbranches]
            if previous is not None:
                current = self._align_subspace(previous, current)
            states[index] = current
            blocks[index] = current.T @ h_env @ current
            previous = current
        return states, 0.5 * (blocks + np.swapaxes(blocks, 1, 2))

    def narg_effective_hamiltonian(self, nbranches: int = 1):
        """Build ``H_eff`` by conditional-state projection.

        ``H_eff[(i,a),(j,b)] = T_q[i,j] <Omega_i^a|Omega_j^b>
        + delta_ij <Omega_i^a|H_env(q_i)|Omega_i^b>``.
        """
        states, blocks = self.conditional_environment_states(nbranches)
        nbranches = int(nbranches)
        dimension = self.active_npoints * nbranches
        hamiltonian = np.zeros((dimension, dimension), dtype=float)
        dressing = np.empty(
            (self.active_npoints, nbranches, self.active_npoints, nbranches),
            dtype=float,
        )
        for i in range(self.active_npoints):
            for j in range(self.active_npoints):
                overlap = states[i].T @ states[j]
                dressing[i, :, j, :] = overlap
                block = self.active_kinetic[i, j] * overlap
                if i == j:
                    block = block + blocks[i]
                rows = slice(i * nbranches, (i + 1) * nbranches)
                cols = slice(j * nbranches, (j + 1) * nbranches)
                hamiltonian[rows, cols] = block
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
        effective_energies = np.linalg.eigvalsh(hamiltonian)
        return Phi4NARGEffectiveHamiltonianResult(
            hamiltonian=hamiltonian,
            active_grid=self.active_grid.copy(),
            environment_grid=self.environment_grid.copy(),
            active_kinetic=self.active_kinetic.copy(),
            conditional_states=states.copy(),
            conditional_blocks=blocks.copy(),
            kinetic_dressing=dressing.copy(),
            effective_energies=effective_energies,
            exact_energies=self.exact_energies(min(effective_energies.size, self.active_npoints * self.environment_npoints)),
            nbranches=nbranches,
        )


@dataclass
class Phi4PeriodicSincNARGResult:
    """Momentum-shell NARG ``H_eff`` for periodic sinc-DVR ``phi^4``."""

    hamiltonian: np.ndarray
    active_configs: np.ndarray
    environment_configs: np.ndarray
    active_modes: np.ndarray
    environment_modes: np.ndarray
    mode_labels: list
    mode_wave_numbers: np.ndarray
    real_space_transform: np.ndarray
    active_kinetic: np.ndarray
    conditional_states: np.ndarray
    conditional_blocks: np.ndarray
    kinetic_dressing: np.ndarray
    effective_energies: np.ndarray
    exact_energies: np.ndarray
    nbranches: int


class Phi4PeriodicSincNARG:
    """Periodic sinc-DVR ``phi^4`` with NARG split in real Fourier modes."""

    def __init__(
        self,
        *,
        spatial_npoints: int = 4,
        length: float = 6.0,
        amplitude_npoints: int = 5,
        field_range: float = 4.5,
        mass2: float = 0.5,
        coupling: float = 0.8,
        active_mode_count: int = 1,
        active_modes=None,
    ):
        self.spatial_npoints = int(spatial_npoints)
        self.length = float(length)
        self.amplitude_npoints = int(amplitude_npoints)
        self.field_range = float(field_range)
        self.mass2 = float(mass2)
        self.coupling = float(coupling)
        if self.spatial_npoints < 1 or self.amplitude_npoints < 1:
            raise ValueError("DVR point counts must be positive.")
        if self.length <= 0 or self.field_range <= 0:
            raise ValueError("length and field_range must be positive.")
        if self.coupling < 0:
            raise ValueError("coupling must be nonnegative.")

        self.x, self.x_weights = periodic_sinc_grid(self.spatial_npoints, self.length)
        self.dx = self.length / self.spatial_npoints
        (
            self.real_space_transform,
            self.mode_wave_numbers,
            self.mode_labels,
        ) = periodic_real_fourier_transform(self.spatial_npoints, self.length)
        self.mode_omega2 = self.mass2 + self.mode_wave_numbers * self.mode_wave_numbers

        if active_modes is None:
            active_modes = np.arange(int(active_mode_count), dtype=int)
        active_modes = np.asarray(active_modes, dtype=int)
        if active_modes.ndim != 1 or active_modes.size < 1:
            raise ValueError("active_modes must be a nonempty one-dimensional list.")
        if np.any(active_modes < 0) or np.any(active_modes >= self.spatial_npoints):
            raise ValueError("active mode indices must fit the Fourier mode count.")
        if np.unique(active_modes).size != active_modes.size:
            raise ValueError("active mode indices must be unique.")
        self.active_modes = active_modes.copy()
        self.environment_modes = np.asarray(
            [mode for mode in range(self.spatial_npoints) if mode not in set(active_modes.tolist())],
            dtype=int,
        )

        length_q = 2.0 * self.field_range
        grid, _ = sine_dvr_grid(self.amplitude_npoints, length_q)
        self.amplitude_grid = grid - self.field_range
        self.amplitude_kinetic = sine_dvr_kinetic_matrix(self.amplitude_npoints, length_q)
        self.active_configs = self._product_configs(self.active_modes.size)
        self.environment_configs = self._product_configs(self.environment_modes.size)
        self.active_kinetic = self._product_kinetic(self.active_modes.size)
        self.environment_kinetic = self._product_kinetic(self.environment_modes.size)
        self._full_hamiltonian_cache = None
        self._exact_energies_cache = None
        self._exact_eigensystem_cache = None

    def _product_configs(self, nmodes: int):
        nmodes = int(nmodes)
        if nmodes == 0:
            return np.zeros((1, 0), dtype=float)
        mesh = np.meshgrid(*([self.amplitude_grid] * nmodes), indexing="ij")
        return np.stack([axis.reshape(-1) for axis in mesh], axis=1)

    def _product_kinetic(self, nmodes: int):
        nmodes = int(nmodes)
        if nmodes == 0:
            return np.zeros((1, 1), dtype=float)
        identity = np.eye(self.amplitude_npoints)
        dimension = self.amplitude_npoints**nmodes
        kinetic = np.zeros((dimension, dimension), dtype=float)
        for mode in range(nmodes):
            operators = [identity] * nmodes
            operators[mode] = self.amplitude_kinetic
            kinetic += _kron_all(operators)
        return 0.5 * (kinetic + kinetic.T)

    def _combine_mode_configs(self, active_config, environment_configs):
        environment_configs = np.asarray(environment_configs, dtype=float)
        if environment_configs.ndim == 1:
            environment_configs = environment_configs.reshape(1, -1)
        combined = np.zeros((environment_configs.shape[0], self.spatial_npoints), dtype=float)
        combined[:, self.active_modes] = np.asarray(active_config, dtype=float)
        combined[:, self.environment_modes] = environment_configs
        return combined

    def potential_from_modes(self, mode_configs):
        mode_configs = np.asarray(mode_configs, dtype=float)
        if mode_configs.ndim == 1:
            mode_configs = mode_configs.reshape(1, -1)
        if mode_configs.shape[1] != self.spatial_npoints:
            raise ValueError("mode_configs must have shape (nconfigs, spatial_npoints).")
        free = 0.5 * np.sum(self.mode_omega2[None, :] * mode_configs * mode_configs, axis=1)
        site_fields = mode_configs @ self.real_space_transform.T
        quartic = self.coupling * np.sum(site_fields**4, axis=1) / (24.0 * self.dx)
        return free + quartic

    def conditional_environment_hamiltonian(self, active_config):
        combined = self._combine_mode_configs(active_config, self.environment_configs)
        potential = self.potential_from_modes(combined)
        return self.environment_kinetic + np.diag(potential)

    def full_mode_configs(self):
        return self._product_configs(self.spatial_npoints)

    def full_hamiltonian_matrix(self):
        if self._full_hamiltonian_cache is None:
            kinetic = self._product_kinetic(self.spatial_npoints)
            potential = self.potential_from_modes(self.full_mode_configs())
            hamiltonian = kinetic + np.diag(potential)
            self._full_hamiltonian_cache = 0.5 * (hamiltonian + hamiltonian.T)
        return self._full_hamiltonian_cache.copy()

    def exact_energies(self, nroots: int | None = None):
        if self._exact_eigensystem_cache is not None:
            values = self._exact_eigensystem_cache[0]
        elif self._exact_energies_cache is None:
            self._exact_energies_cache = np.linalg.eigvalsh(self.full_hamiltonian_matrix())
            values = self._exact_energies_cache
        else:
            values = self._exact_energies_cache
        if nroots is None:
            return values.copy()
        return values[: int(nroots)].copy()

    def exact_eigensystem(self, nroots: int | None = None):
        if self._exact_eigensystem_cache is None:
            self._exact_eigensystem_cache = np.linalg.eigh(self.full_hamiltonian_matrix())
            self._exact_energies_cache = self._exact_eigensystem_cache[0]
        values, vectors = self._exact_eigensystem_cache
        if nroots is None:
            return values.copy(), vectors.copy()
        nroots = int(nroots)
        return values[:nroots].copy(), vectors[:, :nroots].copy()

    def free_analytic_ground_energy(self):
        """Continuum-coordinate free-theory ground energy at this cutoff."""
        return 0.5 * float(np.sum(np.sqrt(self.mode_omega2)))

    def free_analytic_gap(self):
        """Lowest free oscillator excitation energy at this cutoff."""
        return float(np.min(np.sqrt(self.mode_omega2)))

    def weak_coupling_first_order_ground_energy(self):
        """First-order perturbative ground energy for the lattice ``phi^4`` term."""
        omega = np.sqrt(self.mode_omega2)
        mode_variance = 0.5 / omega
        site_variance = (self.real_space_transform * self.real_space_transform) @ mode_variance
        correction = self.coupling * float(np.sum(site_variance * site_variance)) / (8.0 * self.dx)
        return self.free_analytic_ground_energy() + correction

    def z2_parity_operator(self):
        """Global ``phi -> -phi`` parity operator in the full amplitude DVR basis."""
        one_mode = np.zeros((self.amplitude_npoints, self.amplitude_npoints), dtype=float)
        for index in range(self.amplitude_npoints):
            one_mode[self.amplitude_npoints - index - 1, index] = 1.0
        return _kron_all([one_mode] * self.spatial_npoints)

    def z2_parity_expectations(self, nroots: int = 6):
        _, vectors = self.exact_eigensystem(nroots=nroots)
        parity = self.z2_parity_operator()
        return np.asarray(
            [np.vdot(vectors[:, root], parity @ vectors[:, root]).real for root in range(vectors.shape[1])],
            dtype=float,
        )

    def field_moment_expectations(self, power: int = 2, nroots: int = 1):
        """Spatial average of ``<phi(x)^power>`` for exact eigenstates."""
        power = int(power)
        if power < 1:
            raise ValueError("power must be positive.")
        _, vectors = self.exact_eigensystem(nroots=nroots)
        site_fields = self.full_mode_configs() @ self.real_space_transform.T
        diagonal = np.mean(site_fields**power, axis=1)
        probabilities = np.abs(vectors) ** 2
        return probabilities.T @ diagonal

    def conditional_environment_states(self, nbranches: int = 1):
        nbranches = int(nbranches)
        env_dim = self.environment_configs.shape[0]
        if nbranches < 1 or nbranches > env_dim:
            raise ValueError("nbranches must be between 1 and the environment Hilbert dimension.")
        active_dim = self.active_configs.shape[0]
        states = np.empty((active_dim, env_dim, nbranches), dtype=float)
        blocks = np.empty((active_dim, nbranches, nbranches), dtype=float)
        previous = None
        for index, active_config in enumerate(self.active_configs):
            h_env = self.conditional_environment_hamiltonian(active_config)
            _, vectors = np.linalg.eigh(h_env)
            current = vectors[:, :nbranches]
            if previous is not None:
                current = Phi4TwoSiteNARG._align_subspace(previous, current)
            states[index] = current
            blocks[index] = current.T @ h_env @ current
            previous = current
        return states, 0.5 * (blocks + np.swapaxes(blocks, 1, 2))

    def narg_effective_hamiltonian(self, nbranches: int = 1):
        """Conditional high-momentum-shell NARG effective Hamiltonian."""
        states, blocks = self.conditional_environment_states(nbranches)
        nbranches = int(nbranches)
        active_dim = self.active_configs.shape[0]
        dimension = active_dim * nbranches
        hamiltonian = np.zeros((dimension, dimension), dtype=float)
        dressing = np.empty((active_dim, nbranches, active_dim, nbranches), dtype=float)
        for i in range(active_dim):
            for j in range(active_dim):
                overlap = states[i].T @ states[j]
                dressing[i, :, j, :] = overlap
                block = self.active_kinetic[i, j] * overlap
                if i == j:
                    block = block + blocks[i]
                rows = slice(i * nbranches, (i + 1) * nbranches)
                cols = slice(j * nbranches, (j + 1) * nbranches)
                hamiltonian[rows, cols] = block
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
        effective_energies = np.linalg.eigvalsh(hamiltonian)
        return Phi4PeriodicSincNARGResult(
            hamiltonian=hamiltonian,
            active_configs=self.active_configs.copy(),
            environment_configs=self.environment_configs.copy(),
            active_modes=self.active_modes.copy(),
            environment_modes=self.environment_modes.copy(),
            mode_labels=list(self.mode_labels),
            mode_wave_numbers=self.mode_wave_numbers.copy(),
            real_space_transform=self.real_space_transform.copy(),
            active_kinetic=self.active_kinetic.copy(),
            conditional_states=states.copy(),
            conditional_blocks=blocks.copy(),
            kinetic_dressing=dressing.copy(),
            effective_energies=effective_energies,
            exact_energies=self.exact_energies(min(effective_energies.size, self.amplitude_npoints**self.spatial_npoints)),
            nbranches=nbranches,
        )


@dataclass
class Phi4MomentumSpaceNARGStepResult:
    """One scalar ``phi^4`` momentum-space NARG coarse-graining step."""

    effective_hamiltonian: Phi4PeriodicSincNARGResult
    fitted_hamiltonian: np.ndarray
    active_configs: np.ndarray
    active_modes: np.ndarray
    environment_modes: np.ndarray
    mode_labels: list
    potential_surface: np.ndarray
    fitted_potential: np.ndarray
    fit_residual: np.ndarray
    coefficients: dict
    rms_error: float
    max_abs_error: float
    nbranches: int
    branch_index: int


class Phi4MomentumSpaceNARG(Phi4PeriodicSincNARG):
    """Public scalar ``phi^4`` lattice-field NARG in momentum space.

    This supported interface uses real Fourier coordinates ordered as
    ``zero, cos(1), sin(1), ...`` with a final Nyquist coordinate for even
    lattices.  The Hamiltonian is the periodic lattice scalar field theory

    ``H = sum_k 1/2 p_k^2 + 1/2 (m^2 + k^2) q_k^2
    + lambda / (24 dx) sum_x phi(x)^4``.

    ``narg_effective_hamiltonian`` performs the conditional projection of the
    environment momentum modes and returns the same data-rich result object as
    ``Phi4PeriodicSincNARG`` for backward-compatible use.
    """

    def _active_phi4_fit_columns(self):
        configs = self.active_configs
        active_k2 = self.mode_wave_numbers[self.active_modes] ** 2
        fixed_gradient = 0.5 * np.sum(active_k2[None, :] * configs * configs, axis=1)
        mass_column = 0.5 * np.sum(configs * configs, axis=1)

        full_configs = np.zeros((configs.shape[0], self.spatial_npoints), dtype=float)
        full_configs[:, self.active_modes] = configs
        active_fields = full_configs @ self.real_space_transform.T
        quartic_column = np.sum(active_fields**4, axis=1) / (24.0 * self.dx)
        design = np.column_stack([np.ones(configs.shape[0]), mass_column, quartic_column])
        return design, fixed_gradient

    def narg_step(self, nbranches: int = 1, *, branch_index: int = 0):
        """Integrate environment modes and fit an active ``phi^4`` Hamiltonian.

        The returned step contains the full conditional ``H_eff`` together with
        a least-squares projection of the selected retained environment branch
        onto the active-mode scalar-field form

        ``c0 + 1/2 (m_eff^2 + k^2) q^2 + lambda_eff / (24 dx) sum_x phi_A(x)^4``.
        """
        effective = self.narg_effective_hamiltonian(nbranches=nbranches)
        branch_index = int(branch_index)
        nbranches = int(nbranches)
        if branch_index < 0 or branch_index >= nbranches:
            raise ValueError("branch_index must select one retained branch.")

        surface = effective.conditional_blocks[:, branch_index, branch_index].copy()
        design, fixed_gradient = self._active_phi4_fit_columns()
        target = surface - fixed_gradient
        values, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted_potential = fixed_gradient + design @ values
        residual = surface - fitted_potential
        fitted_hamiltonian = self.active_kinetic + np.diag(fitted_potential)
        coefficients = {
            "constant": float(values[0]),
            "mass2": float(values[1]),
            "coupling": float(values[2]),
        }
        return Phi4MomentumSpaceNARGStepResult(
            effective_hamiltonian=effective,
            fitted_hamiltonian=0.5 * (fitted_hamiltonian + fitted_hamiltonian.T),
            active_configs=self.active_configs.copy(),
            active_modes=self.active_modes.copy(),
            environment_modes=self.environment_modes.copy(),
            mode_labels=list(self.mode_labels),
            potential_surface=surface,
            fitted_potential=fitted_potential,
            fit_residual=residual,
            coefficients=coefficients,
            rms_error=float(np.sqrt(np.mean(residual * residual))),
            max_abs_error=float(np.max(np.abs(residual))),
            nbranches=nbranches,
            branch_index=branch_index,
        )


Phi4MomentumSpaceNARGResult = Phi4PeriodicSincNARGResult


@dataclass
class Phi4LogShellNARGResult:
    """Log-discretized momentum-shell NARG ``H_eff`` for ``phi^4``."""

    hamiltonian: np.ndarray
    active_configs: np.ndarray
    environment_configs: np.ndarray
    active_modes: np.ndarray
    environment_modes: np.ndarray
    mode_labels: list
    mode_wave_numbers: np.ndarray
    shell_edges: np.ndarray
    shell_widths: np.ndarray
    shell_weights: np.ndarray
    real_space_basis: np.ndarray
    active_kinetic: np.ndarray
    conditional_states: np.ndarray
    conditional_blocks: np.ndarray
    kinetic_dressing: np.ndarray
    effective_energies: np.ndarray
    exact_energies: np.ndarray
    nbranches: int


@dataclass
class Phi4LogShellIterativeNARGResult:
    """Iterative log-shell NARG/NRG growth result."""

    hamiltonian: np.ndarray
    energies: np.ndarray
    kept_basis: np.ndarray
    included_modes: np.ndarray
    mode_configs: np.ndarray
    records: list
    exact_energies: np.ndarray
    kept_dim: int
    direction: str


@dataclass
class Phi4LogShellCoarseGrainResult:
    """One conditional NARG coarse-graining step for log momentum shells."""

    effective_hamiltonian: Phi4LogShellNARGResult
    fitted_hamiltonian: np.ndarray
    active_configs: np.ndarray
    active_modes: np.ndarray
    environment_modes: np.ndarray
    mode_labels: list
    shell_edges: np.ndarray
    retained_shells: int
    integrated_shells: int
    new_cutoff: float
    potential_surface: np.ndarray
    fitted_potential: np.ndarray
    fit_residual: np.ndarray
    coefficients: dict
    rms_error: float
    max_abs_error: float
    nbranches: int
    branch_index: int


@dataclass
class Phi4LogShellCouplingFlowStep:
    """One sampled NARG coupling-flow shell step."""

    shell: int
    integrated_modes: np.ndarray
    retained_modes: np.ndarray
    new_cutoff: float
    coefficients: dict
    sample_count: int
    rms_error: float
    max_abs_error: float
    energy_min: float
    energy_max: float


@dataclass
class Phi4LogShellCouplingFlowResult:
    """Sampled many-shell NARG coupling flow."""

    steps: list
    initial_coefficients: dict
    final_coefficients: dict
    cutoff: float
    final_cutoff: float
    log_factor: float
    retained_shells: int
    amplitude_npoints: int
    field_range: float
    branch_index: int
    spatial_dim: int = 1

    def dimensionless_rows(self, spatial_dim: int | None = None):
        """Return dimensionless running couplings and finite-difference betas."""
        if spatial_dim is None:
            spatial_dim = self.spatial_dim
        spatial_dim = int(spatial_dim)
        rows = [
            {
                "step": -1,
                "shell": None,
                "cutoff": float(self.cutoff),
                "rg_time": 0.0,
                "mass2": float(self.initial_coefficients["mass2"]),
                "coupling": float(self.initial_coefficients["coupling"]),
                "constant": float(self.initial_coefficients.get("constant", 0.0)),
                "gradient_z": float(self.initial_coefficients.get("gradient_z", 1.0)),
                "phi6": float(self.initial_coefficients.get("phi6", 0.0)),
                "phi8": float(self.initial_coefficients.get("phi8", 0.0)),
                "r": float(self.initial_coefficients["mass2"]) / (float(self.cutoff) ** 2),
                "g": float(self.initial_coefficients["coupling"]) / (float(self.cutoff) ** (3 - spatial_dim)),
                "beta_r": np.nan,
                "beta_g": np.nan,
                "rms_error": 0.0,
                "max_abs_error": 0.0,
            }
        ]
        for index, step in enumerate(self.steps):
            cutoff = float(step.new_cutoff)
            coeff = step.coefficients
            rows.append(
                {
                    "step": index,
                    "shell": int(step.shell),
                    "cutoff": cutoff,
                    "rg_time": float(np.log(float(self.cutoff) / cutoff)),
                    "mass2": float(coeff["mass2"]),
                    "coupling": float(coeff["coupling"]),
                    "constant": float(coeff["constant_total"]),
                    "gradient_z": float(coeff.get("gradient_z", 1.0)),
                    "phi6": float(coeff.get("phi6", 0.0)),
                    "phi8": float(coeff.get("phi8", 0.0)),
                    "r": float(coeff["mass2"]) / (cutoff**2),
                    "g": float(coeff["coupling"]) / (cutoff ** (3 - spatial_dim)),
                    "beta_r": np.nan,
                    "beta_g": np.nan,
                    "rms_error": float(step.rms_error),
                    "max_abs_error": float(step.max_abs_error),
                }
            )
        for previous, current in zip(rows[:-1], rows[1:]):
            dl = current["rg_time"] - previous["rg_time"]
            if dl != 0:
                current["beta_r"] = (current["r"] - previous["r"]) / dl
                current["beta_g"] = (current["g"] - previous["g"]) / dl
        return rows


class Phi4LogShellNARG:
    """Log-discretized ``phi^4`` momentum shells with conditional NARG.

    Shell index ``0`` is the UV shell nearest ``cutoff`` and larger shell
    indices move toward the IR.  The field is reconstructed from one
    representative real cos/sin pair per logarithmic shell, with no explicit
    zero mode.  Each representative carries the shell weight
    ``sqrt(Delta k / delta k)`` in the real-space interaction.
    """

    def __init__(
        self,
        *,
        cutoff: float = 4.0,
        log_factor: float = 2.0,
        nshells: int = 2,
        active_shells: int = 1,
        length: float | None = None,
        amplitude_npoints: int = 4,
        field_range: float = 4.5,
        mass2: float = 0.5,
        coupling: float = 0.8,
        quadrature_order: int | None = None,
        build_dense_spaces: bool = True,
    ):
        self.cutoff = float(cutoff)
        self.log_factor = float(log_factor)
        self.nshells = int(nshells)
        self.active_shells = int(active_shells)
        self.amplitude_npoints = int(amplitude_npoints)
        self.field_range = float(field_range)
        self.mass2 = float(mass2)
        self.coupling = float(coupling)
        self.build_dense_spaces = bool(build_dense_spaces)
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if self.log_factor <= 1:
            raise ValueError("log_factor must be greater than one.")
        if self.nshells < 1:
            raise ValueError("nshells must be positive.")
        if self.active_shells < 0 or self.active_shells > self.nshells:
            raise ValueError("active_shells must be between 0 and nshells.")
        if self.amplitude_npoints < 1:
            raise ValueError("amplitude_npoints must be positive.")
        if self.field_range <= 0:
            raise ValueError("field_range must be positive.")
        if self.coupling < 0:
            raise ValueError("coupling must be nonnegative.")

        powers = np.arange(self.nshells + 1, dtype=float)
        self.shell_edges = self.cutoff / (self.log_factor**powers)
        self.shell_widths = self.shell_edges[:-1] - self.shell_edges[1:]
        self.shell_representatives = np.sqrt(self.shell_edges[:-1] * self.shell_edges[1:])
        self.ir_cutoff = float(self.shell_edges[-1])
        self.length = float(length) if length is not None else float(2.0 * np.pi / self.ir_cutoff)
        if self.length <= 0:
            raise ValueError("length must be positive.")
        self.delta_k = 2.0 * np.pi / self.length
        self.shell_weights = np.sqrt(self.shell_widths / self.delta_k)

        self.mode_labels = []
        mode_wave_numbers = []
        mode_weights = []
        shell_index_by_mode = []
        for shell, (k_value, shell_weight) in enumerate(
            zip(self.shell_representatives, self.shell_weights),
        ):
            self.mode_labels.append(("cos", shell))
            mode_wave_numbers.append(float(k_value))
            mode_weights.append(float(shell_weight))
            shell_index_by_mode.append(shell)
            self.mode_labels.append(("sin", shell))
            mode_wave_numbers.append(float(k_value))
            mode_weights.append(float(shell_weight))
            shell_index_by_mode.append(shell)
        self.mode_wave_numbers = np.asarray(mode_wave_numbers, dtype=float)
        self.mode_weights = np.asarray(mode_weights, dtype=float)
        self.shell_index_by_mode = np.asarray(shell_index_by_mode, dtype=int)
        self.nmodes = self.mode_wave_numbers.size
        self.mode_omega2 = self.mass2 + self.mode_wave_numbers * self.mode_wave_numbers

        first_active_shell = self.nshells - self.active_shells
        active_modes = []
        for shell in range(first_active_shell, self.nshells):
            active_modes.extend([2 * shell, 2 * shell + 1])
        self.active_modes = np.asarray(active_modes, dtype=int)
        active_mask = np.zeros(self.nmodes, dtype=bool)
        active_mask[self.active_modes] = True
        self.environment_modes = np.nonzero(~active_mask)[0].astype(int)

        length_q = 2.0 * self.field_range
        grid, _ = sine_dvr_grid(self.amplitude_npoints, length_q)
        self.amplitude_grid = grid - self.field_range
        self.amplitude_kinetic = sine_dvr_kinetic_matrix(self.amplitude_npoints, length_q)
        if self.build_dense_spaces:
            self.active_configs = self._product_configs(self.active_modes.size)
            self.environment_configs = self._product_configs(self.environment_modes.size)
            self.active_kinetic = self._product_kinetic(self.active_modes.size)
            self.environment_kinetic = self._product_kinetic(self.environment_modes.size)
        else:
            self.active_configs = None
            self.environment_configs = None
            self.active_kinetic = None
            self.environment_kinetic = None

        if quadrature_order is None:
            quadrature_order = max(128, 32 * self.nshells)
        self.quadrature_order = int(quadrature_order)
        self.x = self.length * np.arange(self.quadrature_order, dtype=float) / self.quadrature_order
        self.x_weights = np.full(self.quadrature_order, self.length / self.quadrature_order, dtype=float)
        self.real_space_basis = self._real_space_basis_values(self.x)
        self._full_hamiltonian_cache = None
        self._exact_energies_cache = None

    def _require_dense_spaces(self):
        if not self.build_dense_spaces:
            raise ValueError("This operation requires build_dense_spaces=True.")

    def _real_space_basis_values(self, x):
        x = np.asarray(x, dtype=float)
        basis = np.empty((x.size, self.nmodes), dtype=float)
        for mode, (kind, _) in enumerate(self.mode_labels):
            k_value = self.mode_wave_numbers[mode]
            weight = self.mode_weights[mode] * np.sqrt(2.0 / self.length)
            if kind == "cos":
                basis[:, mode] = weight * np.cos(k_value * x)
            elif kind == "sin":
                basis[:, mode] = weight * np.sin(k_value * x)
            else:
                raise ValueError(f"unsupported mode label {kind!r}.")
        return basis

    def _product_configs(self, nmodes: int):
        nmodes = int(nmodes)
        if nmodes == 0:
            return np.zeros((1, 0), dtype=float)
        mesh = np.meshgrid(*([self.amplitude_grid] * nmodes), indexing="ij")
        return np.stack([axis.reshape(-1) for axis in mesh], axis=1)

    def _product_kinetic(self, nmodes: int):
        nmodes = int(nmodes)
        if nmodes == 0:
            return np.zeros((1, 1), dtype=float)
        identity = np.eye(self.amplitude_npoints)
        dimension = self.amplitude_npoints**nmodes
        kinetic = np.zeros((dimension, dimension), dtype=float)
        for mode in range(nmodes):
            operators = [identity] * nmodes
            operators[mode] = self.amplitude_kinetic
            kinetic += _kron_all(operators)
        return 0.5 * (kinetic + kinetic.T)

    def _combine_mode_configs(self, active_config, environment_configs):
        environment_configs = np.asarray(environment_configs, dtype=float)
        if environment_configs.ndim == 1:
            environment_configs = environment_configs.reshape(1, -1)
        combined = np.zeros((environment_configs.shape[0], self.nmodes), dtype=float)
        combined[:, self.active_modes] = np.asarray(active_config, dtype=float)
        combined[:, self.environment_modes] = environment_configs
        return combined

    def potential_from_modes(self, mode_configs):
        mode_configs = np.asarray(mode_configs, dtype=float)
        if mode_configs.ndim == 1:
            mode_configs = mode_configs.reshape(1, -1)
        if mode_configs.shape[1] != self.nmodes:
            raise ValueError("mode_configs must have shape (nconfigs, nmodes).")
        free = 0.5 * np.sum(self.mode_omega2[None, :] * mode_configs * mode_configs, axis=1)
        site_fields = mode_configs @ self.real_space_basis.T
        quartic = self.coupling * (site_fields**4) @ self.x_weights / 24.0
        return free + quartic

    def partial_potential_from_modes(self, mode_configs, mode_indices):
        mode_configs = np.asarray(mode_configs, dtype=float)
        mode_indices = np.asarray(mode_indices, dtype=int)
        if mode_configs.ndim == 1:
            mode_configs = mode_configs.reshape(1, -1)
        if mode_configs.shape[1] != mode_indices.size:
            raise ValueError("mode_configs must match mode_indices.")
        full = np.zeros((mode_configs.shape[0], self.nmodes), dtype=float)
        full[:, mode_indices] = mode_configs
        return self.potential_from_modes(full)

    def conditional_environment_hamiltonian(self, active_config):
        self._require_dense_spaces()
        combined = self._combine_mode_configs(active_config, self.environment_configs)
        potential = self.potential_from_modes(combined)
        return self.environment_kinetic + np.diag(potential)

    def full_mode_configs(self):
        return self._product_configs(self.nmodes)

    def full_hamiltonian_matrix(self):
        self._require_dense_spaces()
        if self._full_hamiltonian_cache is None:
            kinetic = self._product_kinetic(self.nmodes)
            potential = self.potential_from_modes(self.full_mode_configs())
            hamiltonian = kinetic + np.diag(potential)
            self._full_hamiltonian_cache = 0.5 * (hamiltonian + hamiltonian.T)
        return self._full_hamiltonian_cache.copy()

    def exact_energies(self, nroots: int | None = None):
        if self._exact_energies_cache is None:
            self._exact_energies_cache = np.linalg.eigvalsh(self.full_hamiltonian_matrix())
        if nroots is None:
            return self._exact_energies_cache.copy()
        return self._exact_energies_cache[: int(nroots)].copy()

    @staticmethod
    def _align_subspace(previous, current):
        overlap = previous.T @ current
        left, _, right_h = np.linalg.svd(overlap)
        rotation = right_h.T @ left.T
        return current @ rotation

    def conditional_environment_states(self, nbranches: int = 1):
        self._require_dense_spaces()
        nbranches = int(nbranches)
        env_dim = self.environment_configs.shape[0]
        if nbranches < 1 or nbranches > env_dim:
            raise ValueError("nbranches must be between 1 and the environment Hilbert dimension.")
        active_dim = self.active_configs.shape[0]
        states = np.empty((active_dim, env_dim, nbranches), dtype=float)
        blocks = np.empty((active_dim, nbranches, nbranches), dtype=float)
        previous = None
        for index, active_config in enumerate(self.active_configs):
            h_env = self.conditional_environment_hamiltonian(active_config)
            _, vectors = np.linalg.eigh(h_env)
            current = vectors[:, :nbranches]
            if previous is not None:
                current = self._align_subspace(previous, current)
            states[index] = current
            blocks[index] = current.T @ h_env @ current
            previous = current
        return states, 0.5 * (blocks + np.swapaxes(blocks, 1, 2))

    def narg_effective_hamiltonian(self, nbranches: int = 1):
        states, blocks = self.conditional_environment_states(nbranches)
        nbranches = int(nbranches)
        active_dim = self.active_configs.shape[0]
        dimension = active_dim * nbranches
        hamiltonian = np.zeros((dimension, dimension), dtype=float)
        dressing = np.empty((active_dim, nbranches, active_dim, nbranches), dtype=float)
        for i in range(active_dim):
            for j in range(active_dim):
                overlap = states[i].T @ states[j]
                dressing[i, :, j, :] = overlap
                block = self.active_kinetic[i, j] * overlap
                if i == j:
                    block = block + blocks[i]
                rows = slice(i * nbranches, (i + 1) * nbranches)
                cols = slice(j * nbranches, (j + 1) * nbranches)
                hamiltonian[rows, cols] = block
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
        effective_energies = np.linalg.eigvalsh(hamiltonian)
        return Phi4LogShellNARGResult(
            hamiltonian=hamiltonian,
            active_configs=self.active_configs.copy(),
            environment_configs=self.environment_configs.copy(),
            active_modes=self.active_modes.copy(),
            environment_modes=self.environment_modes.copy(),
            mode_labels=list(self.mode_labels),
            mode_wave_numbers=self.mode_wave_numbers.copy(),
            shell_edges=self.shell_edges.copy(),
            shell_widths=self.shell_widths.copy(),
            shell_weights=self.shell_weights.copy(),
            real_space_basis=self.real_space_basis.copy(),
            active_kinetic=self.active_kinetic.copy(),
            conditional_states=states.copy(),
            conditional_blocks=blocks.copy(),
            kinetic_dressing=dressing.copy(),
            effective_energies=effective_energies,
            exact_energies=self.exact_energies(min(effective_energies.size, self.amplitude_npoints**self.nmodes)),
            nbranches=nbranches,
        )

    def _active_phi4_fit_columns(self):
        configs = self.active_configs
        active_k2 = self.mode_wave_numbers[self.active_modes] ** 2
        fixed_gradient = 0.5 * np.sum(active_k2[None, :] * configs * configs, axis=1)
        mass_column = 0.5 * np.sum(configs * configs, axis=1)

        full_configs = np.zeros((configs.shape[0], self.nmodes), dtype=float)
        full_configs[:, self.active_modes] = configs
        active_fields = full_configs @ self.real_space_basis.T
        quartic_column = (active_fields**4) @ self.x_weights / 24.0
        design = np.column_stack([np.ones(configs.shape[0]), mass_column, quartic_column])
        return design, fixed_gradient

    def _potential_from_partial_modes_with_couplings(self, mode_configs, mode_indices, mass2, coupling):
        mode_configs = np.asarray(mode_configs, dtype=float)
        mode_indices = np.asarray(mode_indices, dtype=int)
        mass2 = float(mass2)
        coupling = float(coupling)
        if mode_configs.ndim == 1:
            mode_configs = mode_configs.reshape(1, -1)
        if mode_configs.shape[1] != mode_indices.size:
            raise ValueError("mode_configs must match mode_indices.")
        free = 0.5 * np.sum(
            (mass2 + self.mode_wave_numbers[mode_indices] ** 2)[None, :] * mode_configs * mode_configs,
            axis=1,
        )
        site_fields = mode_configs @ self.real_space_basis[:, mode_indices].T
        quartic = coupling * (site_fields**4) @ self.x_weights / 24.0
        return free + quartic

    def _coupling_flow_samples(
        self,
        mode_indices,
        amplitudes=None,
        *,
        sample_rule: str = "amplitudes",
        sample_order: int = 3,
    ):
        mode_indices = np.asarray(mode_indices, dtype=int)
        nmodes = mode_indices.size
        if nmodes == 0:
            return np.zeros((1, 0), dtype=float)
        sample_rule = str(sample_rule).lower().replace("-", "_")
        if sample_rule in {"quadrature", "sparse_quadrature", "sparse"}:
            sample_order = int(sample_order)
            if sample_order < 2:
                raise ValueError("sample_order must be at least 2 for quadrature sampling.")
            nodes, _ = leggauss(sample_order)
            nodes = self.field_range * nodes
            samples = [np.zeros(nmodes, dtype=float)]

            for column in range(nmodes):
                for node in nodes:
                    sample = np.zeros(nmodes, dtype=float)
                    sample[column] = node
                    samples.append(sample)

            shell_to_columns = {}
            for column, mode in enumerate(mode_indices):
                shell_to_columns.setdefault(int(self.shell_index_by_mode[mode]), []).append(column)
            for columns in shell_to_columns.values():
                if len(columns) >= 2:
                    for left in nodes:
                        for right in nodes:
                            sample = np.zeros(nmodes, dtype=float)
                            sample[columns[0]] = left
                            sample[columns[1]] = right
                            samples.append(sample)

            shells = sorted(shell_to_columns)
            for left_shell, right_shell in zip(shells[:-1], shells[1:]):
                left_column = shell_to_columns[left_shell][0]
                right_column = shell_to_columns[right_shell][0]
                for node in nodes:
                    sample = np.zeros(nmodes, dtype=float)
                    sample[left_column] = node / np.sqrt(2.0)
                    sample[right_column] = node / np.sqrt(2.0)
                    samples.append(sample)
            return np.unique(np.asarray(samples, dtype=float), axis=0)

        if sample_rule not in {"amplitudes", "amplitude", "radial"}:
            raise ValueError("sample_rule must be 'amplitudes' or 'quadrature'.")
        if amplitudes is None:
            amplitudes = (0.25 * self.field_range, 0.5 * self.field_range)
        amplitudes = tuple(float(value) for value in amplitudes)
        samples = [np.zeros(nmodes, dtype=float)]

        for amplitude in amplitudes:
            for column in range(nmodes):
                sample = np.zeros(nmodes, dtype=float)
                sample[column] = amplitude
                samples.append(sample)

            shell_to_columns = {}
            for column, mode in enumerate(mode_indices):
                shell_to_columns.setdefault(int(self.shell_index_by_mode[mode]), []).append(column)
            for columns in shell_to_columns.values():
                if len(columns) >= 2:
                    sample = np.zeros(nmodes, dtype=float)
                    sample[columns[0]] = amplitude / np.sqrt(2.0)
                    sample[columns[1]] = amplitude / np.sqrt(2.0)
                    samples.append(sample)

            shells = sorted(shell_to_columns)
            for left_shell, right_shell in zip(shells[:-1], shells[1:]):
                sample = np.zeros(nmodes, dtype=float)
                sample[shell_to_columns[left_shell][0]] = amplitude / np.sqrt(2.0)
                sample[shell_to_columns[right_shell][0]] = amplitude / np.sqrt(2.0)
                samples.append(sample)

        return np.unique(np.asarray(samples, dtype=float), axis=0)

    def _coupling_flow_fit_terms(
        self,
        samples,
        retained_modes,
        *,
        max_power: int = 4,
        fit_gradient: bool = False,
    ):
        samples = np.asarray(samples, dtype=float)
        retained_modes = np.asarray(retained_modes, dtype=int)
        max_power = int(max_power)
        if max_power not in {4, 6, 8}:
            raise ValueError("max_power must be 4, 6, or 8.")

        if retained_modes.size:
            gradient_column = 0.5 * np.sum(
                self.mode_wave_numbers[retained_modes][None, :] ** 2 * samples * samples,
                axis=1,
            )
            mass_column = 0.5 * np.sum(samples * samples, axis=1)
            retained_fields = samples @ self.real_space_basis[:, retained_modes].T
        else:
            gradient_column = np.zeros(samples.shape[0], dtype=float)
            mass_column = np.zeros(samples.shape[0], dtype=float)
            retained_fields = np.zeros((samples.shape[0], self.x.size), dtype=float)

        columns = [np.ones(samples.shape[0], dtype=float)]
        names = ["constant"]
        fixed = np.zeros(samples.shape[0], dtype=float)
        if fit_gradient:
            columns.append(gradient_column)
            names.append("gradient_z")
        else:
            fixed = fixed + gradient_column
        columns.append(mass_column)
        names.append("mass2")
        columns.append((retained_fields**4) @ self.x_weights / 24.0)
        names.append("coupling")
        if max_power >= 6:
            columns.append((retained_fields**6) @ self.x_weights / 720.0)
            names.append("phi6")
        if max_power >= 8:
            columns.append((retained_fields**8) @ self.x_weights / 40320.0)
            names.append("phi8")
        return np.column_stack(columns), names, fixed

    def narg_coupling_flow(
        self,
        *,
        retained_shells: int = 1,
        branch_index: int = 0,
        amplitudes=None,
        spatial_dim: int = 1,
        sample_rule: str = "amplitudes",
        sample_order: int = 3,
        max_power: int = 4,
        fit_gradient: bool = False,
    ):
        """Run a sampled many-shell NARG coarse-graining flow.

        Each step integrates the current UV cos/sin shell by diagonalizing its
        conditional Hamiltonian over sampled configurations of the remaining
        lower shells.  The resulting adiabatic surface is projected back onto a
        scalar ``phi^4`` form, giving running ``mass2`` and ``coupling``.
        """
        retained_shells = int(retained_shells)
        branch_index = int(branch_index)
        if retained_shells < 1 or retained_shells > self.nshells:
            raise ValueError("retained_shells must be between 1 and nshells.")
        if branch_index < 0:
            raise ValueError("branch_index must be nonnegative.")
        if fit_gradient and retained_shells < 2:
            raise ValueError("fit_gradient requires at least two retained shells.")

        mass2 = float(self.mass2)
        coupling = float(self.coupling)
        constant = 0.0
        steps = []
        stop_shell = self.nshells - retained_shells
        for shell in range(stop_shell):
            integrated_modes = np.asarray([2 * shell, 2 * shell + 1], dtype=int)
            retained_modes = np.arange(2 * (shell + 1), self.nmodes, dtype=int)
            shell_configs = self._product_configs(integrated_modes.size)
            shell_kinetic = self._product_kinetic(integrated_modes.size)
            shell_dim = shell_configs.shape[0]
            if branch_index >= shell_dim:
                raise ValueError("branch_index exceeds the one-shell Hilbert dimension.")

            samples = self._coupling_flow_samples(
                retained_modes,
                amplitudes=amplitudes,
                sample_rule=sample_rule,
                sample_order=sample_order,
            )
            energies = np.empty(samples.shape[0], dtype=float)
            combined_modes = np.concatenate([integrated_modes, retained_modes])
            for sample_index, retained_config in enumerate(samples):
                combined = np.hstack(
                    [
                        shell_configs,
                        np.broadcast_to(retained_config, (shell_dim, retained_config.size)),
                    ]
                )
                potential = self._potential_from_partial_modes_with_couplings(
                    combined,
                    combined_modes,
                    mass2,
                    coupling,
                )
                hamiltonian = shell_kinetic + np.diag(potential)
                values = np.linalg.eigvalsh(0.5 * (hamiltonian + hamiltonian.T))
                energies[sample_index] = values[branch_index]

            design, names, fixed = self._coupling_flow_fit_terms(
                samples,
                retained_modes,
                max_power=max_power,
                fit_gradient=fit_gradient,
            )
            target = energies - fixed
            values, *_ = np.linalg.lstsq(design, target, rcond=None)
            fitted = fixed + design @ values
            residual = energies - fitted

            step_coefficients = {name: float(value) for name, value in zip(names, values)}
            constant += float(step_coefficients["constant"])
            mass2 = float(step_coefficients["mass2"])
            coupling = float(step_coefficients["coupling"])
            if not fit_gradient:
                step_coefficients["gradient_z"] = 1.0
            step_coefficients["constant_step"] = float(step_coefficients["constant"])
            step_coefficients["constant_total"] = float(constant)
            steps.append(
                Phi4LogShellCouplingFlowStep(
                    shell=int(shell),
                    integrated_modes=integrated_modes.copy(),
                    retained_modes=retained_modes.copy(),
                    new_cutoff=float(self.shell_edges[shell + 1]),
                    coefficients=step_coefficients,
                    sample_count=int(samples.shape[0]),
                    rms_error=float(np.sqrt(np.mean(residual * residual))),
                    max_abs_error=float(np.max(np.abs(residual))),
                    energy_min=float(np.min(energies)),
                    energy_max=float(np.max(energies)),
                )
            )

        final_cutoff = float(self.shell_edges[stop_shell])
        final_coefficients = {
            "constant": float(constant),
            "mass2": float(mass2),
            "coupling": float(coupling),
        }
        if steps:
            last = steps[-1].coefficients
            for name in ("gradient_z", "phi6", "phi8"):
                if name in last:
                    final_coefficients[name] = float(last[name])
        return Phi4LogShellCouplingFlowResult(
            steps=steps,
            initial_coefficients={
                "constant": 0.0,
                "mass2": float(self.mass2),
                "coupling": float(self.coupling),
                "gradient_z": 1.0,
                "phi6": 0.0,
                "phi8": 0.0,
            },
            final_coefficients=final_coefficients,
            cutoff=float(self.cutoff),
            final_cutoff=final_cutoff,
            log_factor=float(self.log_factor),
            retained_shells=retained_shells,
            amplitude_npoints=int(self.amplitude_npoints),
            field_range=float(self.field_range),
            branch_index=branch_index,
            spatial_dim=int(spatial_dim),
        )

    def sampled_vs_dense_coarse_grain(
        self,
        *,
        retained_shells: int | None = None,
        nbranches: int = 1,
        branch_index: int = 0,
        amplitudes=None,
        sample_rule: str = "amplitudes",
        sample_order: int = 3,
        max_power: int = 4,
        fit_gradient: bool = False,
    ):
        """Compare sampled one-shell NARG fitting against dense conditional NARG."""
        if retained_shells is None:
            retained_shells = self.nshells - 1
        retained_shells = int(retained_shells)
        if self.nshells - retained_shells != 1:
            raise ValueError("sampled_vs_dense_coarse_grain compares one integrated UV shell.")
        if not self.build_dense_spaces:
            dense = Phi4LogShellNARG(
                cutoff=self.cutoff,
                log_factor=self.log_factor,
                nshells=self.nshells,
                active_shells=retained_shells,
                length=self.length,
                amplitude_npoints=self.amplitude_npoints,
                field_range=self.field_range,
                mass2=self.mass2,
                coupling=self.coupling,
                quadrature_order=self.quadrature_order,
                build_dense_spaces=True,
            )
        else:
            dense = self
        exact = dense.narg_coarse_grain_step(
            nbranches=nbranches,
            retained_shells=retained_shells,
            branch_index=branch_index,
        )
        sampled_model = Phi4LogShellNARG(
            cutoff=self.cutoff,
            log_factor=self.log_factor,
            nshells=self.nshells,
            active_shells=0,
            length=self.length,
            amplitude_npoints=self.amplitude_npoints,
            field_range=self.field_range,
            mass2=self.mass2,
            coupling=self.coupling,
            quadrature_order=self.quadrature_order,
            build_dense_spaces=False,
        )
        sampled = sampled_model.narg_coupling_flow(
            retained_shells=retained_shells,
            branch_index=branch_index,
            amplitudes=amplitudes,
            sample_rule=sample_rule,
            sample_order=sample_order,
            max_power=max_power,
            fit_gradient=fit_gradient,
        )
        sampled_coeff = sampled.steps[0].coefficients
        rows = {}
        for name in ("mass2", "coupling"):
            rows[name] = {
                "dense": float(exact.coefficients[name]),
                "sampled": float(sampled_coeff[name]),
                "abs_error": float(abs(exact.coefficients[name] - sampled_coeff[name])),
            }
        rows["constant"] = {
            "dense": float(exact.coefficients["constant"]),
            "sampled": float(sampled_coeff["constant_step"]),
            "abs_error": float(abs(exact.coefficients["constant"] - sampled_coeff["constant_step"])),
        }
        return {
            "dense": exact,
            "sampled": sampled,
            "coefficients": rows,
            "dense_rms_error": float(exact.rms_error),
            "sampled_rms_error": float(sampled.steps[0].rms_error),
        }

    def narg_coupling_flow_scan(
        self,
        *,
        amplitude_npoints_values=None,
        quadrature_orders=None,
        amplitude_sets=None,
        sample_rules=None,
        sample_orders=None,
        log_factors=None,
        max_powers=None,
        fit_gradient_values=None,
        retained_shells: int = 1,
        branch_index: int = 0,
        spatial_dim: int = 1,
    ):
        """Convergence scan for the sampled many-shell coupling flow."""
        if amplitude_npoints_values is None:
            amplitude_npoints_values = [self.amplitude_npoints]
        if quadrature_orders is None:
            quadrature_orders = [self.quadrature_order]
        if amplitude_sets is None:
            amplitude_sets = [None]
        if sample_rules is None:
            sample_rules = ["amplitudes"]
        if sample_orders is None:
            sample_orders = [3]
        if log_factors is None:
            log_factors = [self.log_factor]
        if max_powers is None:
            max_powers = [4]
        if fit_gradient_values is None:
            fit_gradient_values = [False]

        rows = []
        for amplitude_npoints in amplitude_npoints_values:
            for quadrature_order in quadrature_orders:
                for amplitude_set in amplitude_sets:
                    for sample_rule in sample_rules:
                        for sample_order in sample_orders:
                            for log_factor in log_factors:
                                for max_power in max_powers:
                                    for fit_gradient in fit_gradient_values:
                                        model = Phi4LogShellNARG(
                                            cutoff=self.cutoff,
                                            log_factor=float(log_factor),
                                            nshells=self.nshells,
                                            active_shells=0,
                                            amplitude_npoints=int(amplitude_npoints),
                                            field_range=self.field_range,
                                            mass2=self.mass2,
                                            coupling=self.coupling,
                                            quadrature_order=int(quadrature_order),
                                            build_dense_spaces=False,
                                        )
                                        flow = model.narg_coupling_flow(
                                            retained_shells=retained_shells,
                                            branch_index=branch_index,
                                            amplitudes=amplitude_set,
                                            spatial_dim=spatial_dim,
                                            sample_rule=sample_rule,
                                            sample_order=sample_order,
                                            max_power=max_power,
                                            fit_gradient=fit_gradient,
                                        )
                                        rows.append(
                                            {
                                                "amplitude_npoints": int(amplitude_npoints),
                                                "quadrature_order": int(quadrature_order),
                                                "amplitudes": amplitude_set,
                                                "sample_rule": str(sample_rule),
                                                "sample_order": int(sample_order),
                                                "log_factor": float(log_factor),
                                                "max_power": int(max_power),
                                                "fit_gradient": bool(fit_gradient),
                                                "final_cutoff": float(flow.final_cutoff),
                                                "mass2": float(flow.final_coefficients["mass2"]),
                                                "coupling": float(flow.final_coefficients["coupling"]),
                                                "gradient_z": float(flow.final_coefficients.get("gradient_z", 1.0)),
                                                "phi6": float(flow.final_coefficients.get("phi6", 0.0)),
                                                "phi8": float(flow.final_coefficients.get("phi8", 0.0)),
                                                "constant": float(flow.final_coefficients["constant"]),
                                                "max_rms_error": float(
                                                    max(step.rms_error for step in flow.steps) if flow.steps else 0.0
                                                ),
                                                "max_abs_error": float(
                                                    max(step.max_abs_error for step in flow.steps) if flow.steps else 0.0
                                                ),
                                            }
                                        )
        return rows

    def narg_fixed_point_scan(
        self,
        mass2_values,
        coupling_values,
        *,
        retained_shells: int = 1,
        branch_index: int = 0,
        amplitudes=None,
        sample_rule: str = "quadrature",
        sample_order: int = 3,
        max_power: int = 6,
        fit_gradient: bool = True,
        spatial_dim: int = 1,
    ):
        """Coarse fixed-point diagnostic from final dimensionless beta norm."""
        rows = []
        for mass2 in mass2_values:
            for coupling in coupling_values:
                model = Phi4LogShellNARG(
                    cutoff=self.cutoff,
                    log_factor=self.log_factor,
                    nshells=self.nshells,
                    active_shells=0,
                    amplitude_npoints=self.amplitude_npoints,
                    field_range=self.field_range,
                    mass2=float(mass2),
                    coupling=float(coupling),
                    quadrature_order=self.quadrature_order,
                    build_dense_spaces=False,
                )
                flow = model.narg_coupling_flow(
                    retained_shells=retained_shells,
                    branch_index=branch_index,
                    amplitudes=amplitudes,
                    spatial_dim=spatial_dim,
                    sample_rule=sample_rule,
                    sample_order=sample_order,
                    max_power=max_power,
                    fit_gradient=fit_gradient,
                )
                beta_rows = flow.dimensionless_rows(spatial_dim=spatial_dim)
                final = beta_rows[-1]
                beta_norm = float(np.hypot(final["beta_r"], final["beta_g"]))
                rows.append(
                    {
                        "initial_mass2": float(mass2),
                        "initial_coupling": float(coupling),
                        "final_mass2": float(flow.final_coefficients["mass2"]),
                        "final_coupling": float(flow.final_coefficients["coupling"]),
                        "final_r": float(final["r"]),
                        "final_g": float(final["g"]),
                        "beta_r": float(final["beta_r"]),
                        "beta_g": float(final["beta_g"]),
                        "beta_norm": beta_norm,
                        "max_rms_error": float(max(step.rms_error for step in flow.steps) if flow.steps else 0.0),
                    }
                )
        rows.sort(key=lambda item: item["beta_norm"])
        return rows

    def narg_coarse_grain_step(
        self,
        nbranches: int = 1,
        *,
        retained_shells: int | None = None,
        branch_index: int = 0,
    ):
        """Integrate UV shells conditionally and fit the retained shell theory.

        Unlike ``iterative_shell_narg``, this is a NARG coarse-graining step:
        the retained lower-momentum shells remain in a coordinate basis, while
        UV shell states are solved conditionally for each retained
        configuration and projected into ``H_eff``.
        """
        if retained_shells is None:
            retained_shells = max(self.nshells - 1, 1)
        retained_shells = int(retained_shells)
        if retained_shells < 1 or retained_shells > self.nshells:
            raise ValueError("retained_shells must be between 1 and nshells.")

        model = self
        if self.active_shells != retained_shells:
            model = Phi4LogShellNARG(
                cutoff=self.cutoff,
                log_factor=self.log_factor,
                nshells=self.nshells,
                active_shells=retained_shells,
                length=self.length,
                amplitude_npoints=self.amplitude_npoints,
                field_range=self.field_range,
                mass2=self.mass2,
                coupling=self.coupling,
                quadrature_order=self.quadrature_order,
            )

        nbranches = int(nbranches)
        branch_index = int(branch_index)
        if branch_index < 0 or branch_index >= nbranches:
            raise ValueError("branch_index must select one retained branch.")

        effective = model.narg_effective_hamiltonian(nbranches=nbranches)
        surface = effective.conditional_blocks[:, branch_index, branch_index].copy()
        design, fixed_gradient = model._active_phi4_fit_columns()
        target = surface - fixed_gradient
        values, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted_potential = fixed_gradient + design @ values
        residual = surface - fitted_potential
        fitted_hamiltonian = model.active_kinetic + np.diag(fitted_potential)
        first_retained_shell = model.nshells - retained_shells
        coefficients = {
            "constant": float(values[0]),
            "mass2": float(values[1]),
            "coupling": float(values[2]),
        }
        return Phi4LogShellCoarseGrainResult(
            effective_hamiltonian=effective,
            fitted_hamiltonian=0.5 * (fitted_hamiltonian + fitted_hamiltonian.T),
            active_configs=model.active_configs.copy(),
            active_modes=model.active_modes.copy(),
            environment_modes=model.environment_modes.copy(),
            mode_labels=list(model.mode_labels),
            shell_edges=model.shell_edges.copy(),
            retained_shells=retained_shells,
            integrated_shells=model.nshells - retained_shells,
            new_cutoff=float(model.shell_edges[first_retained_shell]),
            potential_surface=surface,
            fitted_potential=fitted_potential,
            fit_residual=residual,
            coefficients=coefficients,
            rms_error=float(np.sqrt(np.mean(residual * residual))),
            max_abs_error=float(np.max(np.abs(residual))),
            nbranches=nbranches,
            branch_index=branch_index,
        )

    def shell_flow_summary(self, nbranches: int = 2):
        """Move the active cutoff shell by shell and report NARG errors."""
        rows = []
        for active_shells in range(self.nshells + 1):
            toy = Phi4LogShellNARG(
                cutoff=self.cutoff,
                log_factor=self.log_factor,
                nshells=self.nshells,
                active_shells=active_shells,
                length=self.length,
                amplitude_npoints=self.amplitude_npoints,
                field_range=self.field_range,
                mass2=self.mass2,
                coupling=self.coupling,
                quadrature_order=self.quadrature_order,
            )
            branches = min(int(nbranches), toy.environment_configs.shape[0])
            result = toy.narg_effective_hamiltonian(branches)
            exact = float(toy.exact_energies(1)[0])
            rows.append(
                {
                    "active_shells": active_shells,
                    "branches": branches,
                    "dimension": result.hamiltonian.shape[0],
                    "energy": float(result.effective_energies[0]),
                    "exact_energy": exact,
                    "error": float(result.effective_energies[0] - exact),
                }
            )
        return rows

    def _iterative_mode_groups(self, direction: str):
        direction = str(direction).lower().replace("-", "_")
        if direction in {"ir", "ir_to_uv", "low_to_high"}:
            groups = [
                (shell, np.asarray([2 * shell, 2 * shell + 1], dtype=int))
                for shell in range(self.nshells - 1, -1, -1)
            ]
            return direction, groups
        if direction in {"uv", "uv_to_ir", "high_to_low"}:
            groups = [
                (shell, np.asarray([2 * shell, 2 * shell + 1], dtype=int))
                for shell in range(self.nshells)
            ]
            return direction, groups
        raise ValueError("direction must be 'uv_to_ir' or 'ir_to_uv'.")

    @staticmethod
    def _discarded_gap(values, keep: int):
        if keep < len(values):
            return float(values[keep] - values[keep - 1])
        return np.inf

    def iterative_shell_narg(
        self,
        kept_dim: int = 32,
        *,
        max_exact_dim: int = 4096,
        direction: str = "uv_to_ir",
    ):
        """Grow log shells one by one and truncate to ``kept_dim`` states.

        This is the scalable NARG/NRG-style backend: after each cos/sin shell
        supersite is added, the enlarged block Hamiltonian is diagonalized and
        compressed to the lowest retained states.
        """
        kept_dim = int(kept_dim)
        if kept_dim < 1:
            raise ValueError("kept_dim must be positive.")
        max_exact_dim = int(max_exact_dim)

        direction, groups = self._iterative_mode_groups(direction)
        first_shell, first_modes = groups[0]
        included_modes = first_modes.copy()
        configs = self._product_configs(first_modes.size)
        potential = self.partial_potential_from_modes(configs, included_modes)
        hamiltonian = self._product_kinetic(first_modes.size) + np.diag(potential)
        values, vectors = np.linalg.eigh(0.5 * (hamiltonian + hamiltonian.T))
        keep = min(kept_dim, values.size)
        energies = values[:keep]
        kept_basis = vectors[:, :keep]
        block_hamiltonian = np.diag(energies)
        records = [
            {
                "step": 0,
                "shell": int(first_shell),
                "mode_labels": [self.mode_labels[index] for index in first_modes],
                "included_modes": int(included_modes.size),
                "basis_dim": int(configs.shape[0]),
                "projected_dim": int(hamiltonian.shape[0]),
                "kept_dim": int(keep),
                "energy": float(energies[0]),
                "discarded_gap": self._discarded_gap(values, keep),
            }
        ]

        for step, (shell, shell_modes) in enumerate(groups[1:], start=1):
            shell_configs = self._product_configs(shell_modes.size)
            shell_kinetic = self._product_kinetic(shell_modes.size)
            shell_dim = shell_configs.shape[0]
            old_keep = kept_basis.shape[1]

            enlarged = np.kron(block_hamiltonian, np.eye(shell_dim))
            enlarged = enlarged + np.kron(np.eye(old_keep), shell_kinetic)

            for shell_index, shell_config in enumerate(shell_configs):
                combined_configs = np.hstack(
                    [
                        configs,
                        np.broadcast_to(shell_config, (configs.shape[0], shell_config.size)),
                    ]
                )
                combined_modes = np.concatenate([included_modes, shell_modes])
                delta = self.partial_potential_from_modes(combined_configs, combined_modes) - potential
                block = kept_basis.T @ (delta[:, None] * kept_basis)
                rows = slice(shell_index, old_keep * shell_dim, shell_dim)
                enlarged[rows, rows] += block

            enlarged = 0.5 * (enlarged + enlarged.T)
            values, vectors = np.linalg.eigh(enlarged)
            keep = min(kept_dim, values.size)
            energies = values[:keep]
            retained = vectors[:, :keep]

            retained_tensor = retained.reshape(old_keep, shell_dim, keep)
            basis_tensor = np.einsum("ia,asr->isr", kept_basis, retained_tensor, optimize=True)
            kept_basis = basis_tensor.reshape(configs.shape[0] * shell_dim, keep)
            block_hamiltonian = np.diag(energies)

            configs = np.hstack(
                [
                    np.repeat(configs, shell_dim, axis=0),
                    np.tile(shell_configs, (configs.shape[0], 1)),
                ]
            )
            included_modes = np.concatenate([included_modes, shell_modes])
            potential = self.partial_potential_from_modes(configs, included_modes)

            records.append(
                {
                    "step": int(step),
                    "shell": int(shell),
                    "mode_labels": [self.mode_labels[index] for index in shell_modes],
                    "included_modes": int(included_modes.size),
                    "basis_dim": int(configs.shape[0]),
                    "projected_dim": int(enlarged.shape[0]),
                    "kept_dim": int(keep),
                    "energy": float(energies[0]),
                    "discarded_gap": self._discarded_gap(values, keep),
                }
            )

        exact_energies = np.array([], dtype=float)
        full_dim = self.amplitude_npoints**self.nmodes
        if full_dim <= max_exact_dim:
            exact_energies = self.exact_energies(min(energies.size, full_dim))
        return Phi4LogShellIterativeNARGResult(
            hamiltonian=block_hamiltonian.copy(),
            energies=energies.copy(),
            kept_basis=kept_basis.copy(),
            included_modes=included_modes.copy(),
            mode_configs=configs.copy(),
            records=records,
            exact_energies=exact_energies,
            kept_dim=kept_dim,
            direction=direction,
        )

    def iterative_kept_dim_scan(self, kept_dims, *, direction: str = "uv_to_ir", max_exact_dim: int = 4096):
        """Energy convergence scan over retained iterative NARG dimension."""
        rows = []
        for kept_dim in kept_dims:
            result = self.iterative_shell_narg(
                kept_dim=int(kept_dim),
                direction=direction,
                max_exact_dim=max_exact_dim,
            )
            exact = float(result.exact_energies[0]) if result.exact_energies.size else np.nan
            rows.append(
                {
                    "kept_dim": int(kept_dim),
                    "energy": float(result.energies[0]),
                    "exact_energy": exact,
                    "error": float(result.energies[0] - exact) if result.exact_energies.size else np.nan,
                    "largest_projected_dim": int(max(record["projected_dim"] for record in result.records)),
                    "final_discarded_gap": float(result.records[-1]["discarded_gap"]),
                }
            )
        return rows

    def iterative_mode_moments(self, result: Phi4LogShellIterativeNARGResult, power: int = 2):
        """Mode-coordinate moments from an iterative NARG ground state."""
        power = int(power)
        if power < 1:
            raise ValueError("power must be positive.")
        probabilities = np.abs(result.kept_basis[:, 0]) ** 2
        moments = {}
        for column, mode in enumerate(result.included_modes):
            moments[self.mode_labels[int(mode)]] = float(np.sum(probabilities * result.mode_configs[:, column] ** power))
        return moments

    def fit_ir_shell_effective_potential(self, max_power: int = 6):
        """Fit the lowest-k shell surface after integrating UV shells.

        The active sector is the IR cos/sin supersite.  The fit uses the
        conditional ground-state surface for ``r^2 = q_cos^2 + q_sin^2``:

        ``V_eff(r) ~= c0 + 1/2 omega2_eff r^2 + lambda_eff r^4 / 24 + c6 r^6``.
        """
        max_power = int(max_power)
        if max_power not in {4, 6}:
            raise ValueError("max_power must be 4 or 6.")
        toy = Phi4LogShellNARG(
            cutoff=self.cutoff,
            log_factor=self.log_factor,
            nshells=self.nshells,
            active_shells=1,
            length=self.length,
            amplitude_npoints=self.amplitude_npoints,
            field_range=self.field_range,
            mass2=self.mass2,
            coupling=self.coupling,
            quadrature_order=self.quadrature_order,
        )
        _, blocks = toy.conditional_environment_states(nbranches=1)
        radius2 = np.sum(toy.active_configs * toy.active_configs, axis=1)
        values = blocks[:, 0, 0]
        columns = [np.ones_like(radius2), 0.5 * radius2, radius2 * radius2 / 24.0]
        names = ["c0", "omega2_eff", "lambda_eff"]
        if max_power == 6:
            columns.append(radius2**3)
            names.append("c6")
        design = np.column_stack(columns)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        fit = design @ coefficients
        return {
            "active_configs": toy.active_configs.copy(),
            "active_modes": toy.active_modes.copy(),
            "active_mode_labels": [toy.mode_labels[index] for index in toy.active_modes],
            "radius2": radius2.copy(),
            "surface": values.copy(),
            "fit": fit.copy(),
            "coefficients": {name: float(value) for name, value in zip(names, coefficients)},
            "rms_error": float(np.sqrt(np.mean((values - fit) ** 2))),
        }

    def fit_zero_mode_effective_potential(self, max_power: int = 6):
        """Backward-compatible alias for the IR-shell effective potential fit."""
        return self.fit_ir_shell_effective_potential(max_power=max_power)

    def log_factor_scan(self, log_factors, *, kept_dim: int = 16, direction: str = "uv_to_ir"):
        """Scan NARG energies versus logarithmic discretization factor."""
        rows = []
        for log_factor in log_factors:
            toy = Phi4LogShellNARG(
                cutoff=self.cutoff,
                log_factor=float(log_factor),
                nshells=self.nshells,
                active_shells=self.active_shells,
                amplitude_npoints=self.amplitude_npoints,
                field_range=self.field_range,
                mass2=self.mass2,
                coupling=self.coupling,
                quadrature_order=self.quadrature_order,
            )
            result = toy.iterative_shell_narg(kept_dim=kept_dim, direction=direction, max_exact_dim=0)
            rows.append(
                {
                    "log_factor": float(log_factor),
                    "energy": float(result.energies[0]),
                    "ir_cutoff": float(toy.ir_cutoff),
                    "largest_projected_dim": int(max(record["projected_dim"] for record in result.records)),
                }
            )
        return rows

    def cutoff_scan(self, cutoffs, *, kept_dim: int = 16, direction: str = "uv_to_ir"):
        """Scan NARG energies versus UV cutoff at fixed shell count."""
        rows = []
        for cutoff in cutoffs:
            toy = Phi4LogShellNARG(
                cutoff=float(cutoff),
                log_factor=self.log_factor,
                nshells=self.nshells,
                active_shells=self.active_shells,
                amplitude_npoints=self.amplitude_npoints,
                field_range=self.field_range,
                mass2=self.mass2,
                coupling=self.coupling,
                quadrature_order=self.quadrature_order,
            )
            result = toy.iterative_shell_narg(kept_dim=kept_dim, direction=direction, max_exact_dim=0)
            rows.append(
                {
                    "cutoff": float(cutoff),
                    "energy": float(result.energies[0]),
                    "ir_cutoff": float(toy.ir_cutoff),
                    "largest_projected_dim": int(max(record["projected_dim"] for record in result.records)),
                }
            )
        return rows


class Yukawa1DWavefunctionalNARG:
    """A small 1+1D Yukawa wavefunctional NARG regulator.

    The scalar field is represented by a few continuum sine modes,

    ``phi(x) = sum_a q_a f_a(x)``.

    For every continuous field configuration ``q``, the fermionic part is the
    filled negative-energy Slater determinant of the Dirac-like one-body
    Hamiltonian

    ``h_F[phi] = -i sigma_z d_x + sigma_x (m_f + g phi(x))``.

    NARG compression is then the direct Schmidt decomposition of
    ``chi[q] |Omega_F[q]>`` over the scalar-field coordinates.
    """

    def __init__(
        self,
        *,
        length: float = 6.0,
        scalar_mass: float = 0.8,
        fermion_mass: float = 0.4,
        coupling: float = 0.9,
        scalar_modes: int = 2,
        fermion_modes: int = 2,
        fermion_regulator: str = "sine_basis",
        oscillator_nbasis: int = 8,
        field_quadrature_order: int = 12,
        spatial_quadrature_order: int = 160,
        noccupied: int | None = None,
    ):
        self.length = float(length)
        self.scalar_mass = float(scalar_mass)
        self.fermion_mass = float(fermion_mass)
        self.coupling = float(coupling)
        self.scalar_modes = int(scalar_modes)
        self.fermion_modes = int(fermion_modes)
        self.fermion_regulator = str(fermion_regulator).lower().replace("-", "_")
        self.oscillator_nbasis = int(oscillator_nbasis)
        self.field_quadrature_order = int(field_quadrature_order)
        self.spatial_quadrature_order = int(spatial_quadrature_order)
        self.nspin_orbitals = 2 * self.fermion_modes
        self.noccupied = self.fermion_modes if noccupied is None else int(noccupied)
        if self.length <= 0:
            raise ValueError("length must be positive.")
        if self.scalar_mass <= 0:
            raise ValueError("scalar_mass must be positive.")
        if self.scalar_modes < 1 or self.fermion_modes < 1:
            raise ValueError("mode counts must be positive.")
        if self.fermion_regulator == "dvr":
            self.fermion_regulator = "sine_dvr"
        if self.fermion_regulator not in {"sine_basis", "sine_dvr"}:
            raise ValueError("fermion_regulator must be 'sine_basis' or 'sine_dvr'.")
        if self.oscillator_nbasis < 1:
            raise ValueError("oscillator_nbasis must be positive.")
        if self.field_quadrature_order < 1 or self.spatial_quadrature_order < 1:
            raise ValueError("quadrature orders must be positive.")
        if self.noccupied < 0 or self.noccupied > self.nspin_orbitals:
            raise ValueError("noccupied must fit in the spin-orbital space.")

        self.scalar_frequencies = self._scalar_frequencies()
        self.fermion_sector_basis = fixed_particle_basis(self.nspin_orbitals, self.noccupied)
        self._prepare_spatial_matrices()
        self._exact_ground_energy_cache = None

    def _scalar_frequencies(self):
        modes = np.arange(1, self.scalar_modes + 1, dtype=float)
        wave_numbers = np.pi * modes / self.length
        return np.sqrt(self.scalar_mass * self.scalar_mass + wave_numbers * wave_numbers)

    def _prepare_spatial_matrices(self):
        if self.fermion_regulator == "sine_dvr":
            x, weights = sine_dvr_grid(self.fermion_modes, self.length)
            self.x = x
            self.x_weights = weights
            self.scalar_basis_x = sine_basis_values(x, self.scalar_modes, self.length)
            self.fermion_dvr_transform = sine_dvr_transform(self.fermion_modes)
            self.single_electron_keo = sine_dvr_kinetic_matrix(self.fermion_modes, self.length)
            self.derivative_matrix = sine_dvr_derivative_matrix(self.fermion_modes, self.length)
            self.fermion_basis_x = np.eye(self.fermion_modes)
            self.fermion_derivative_x = self.derivative_matrix.copy()
            self.scalar_vertices = [
                np.diag(self.scalar_basis_x[:, mode]).astype(float)
                for mode in range(self.scalar_modes)
            ]
            return

        x, weights = interval_legendre_quadrature(self.spatial_quadrature_order, self.length)
        self.x = x
        self.x_weights = weights
        self.scalar_basis_x = sine_basis_values(x, self.scalar_modes, self.length)
        self.fermion_basis_x = sine_basis_values(x, self.fermion_modes, self.length)
        self.fermion_derivative_x = sine_basis_derivative_values(x, self.fermion_modes, self.length)
        self.derivative_matrix = self.fermion_basis_x.T @ (weights[:, None] * self.fermion_derivative_x)
        modes = np.arange(1, self.fermion_modes + 1, dtype=float)
        self.single_electron_keo = np.diag(0.5 * (np.pi * modes / self.length) ** 2)
        self.scalar_vertices = []
        for mode in range(self.scalar_modes):
            weighted_field = weights * self.scalar_basis_x[:, mode]
            self.scalar_vertices.append(
                self.fermion_basis_x.T @ (weighted_field[:, None] * self.fermion_basis_x)
            )

    def field_quadrature(self):
        """Product Gauss-Hermite quadrature over scalar field coordinates."""
        nodes = []
        weights = []
        for omega in self.scalar_frequencies:
            q, w = hermite_quadrature(self.field_quadrature_order, omega)
            nodes.append(q)
            weights.append(w)
        multi_indices = list(product(*[range(len(mode_nodes)) for mode_nodes in nodes]))
        samples = np.asarray(
            [[nodes[mode][index[mode]] for mode in range(self.scalar_modes)] for index in multi_indices],
            dtype=float,
        )
        sample_weights = np.asarray(
            [
                np.prod([weights[mode][index[mode]] for mode in range(self.scalar_modes)])
                for index in multi_indices
            ],
            dtype=float,
        )
        return samples, sample_weights

    def boson_reference(self, q_samples, widths=None, centers=None):
        q_samples = np.asarray(q_samples, dtype=float)
        widths = self.scalar_frequencies if widths is None else np.asarray(widths, dtype=float)
        centers = np.zeros(self.scalar_modes) if centers is None else np.asarray(centers, dtype=float)
        if widths.shape != (self.scalar_modes,) or centers.shape != (self.scalar_modes,):
            raise ValueError("widths and centers must have shape (scalar_modes,).")
        if np.any(widths <= 0):
            raise ValueError("all Gaussian widths must be positive.")
        values = np.ones(q_samples.shape[0], dtype=float)
        for mode, width in enumerate(widths):
            shifted = q_samples[:, mode] - centers[mode]
            values *= (width / np.pi) ** 0.25 * np.exp(-0.5 * width * shifted * shifted)
        return values

    def fermion_one_body_hamiltonian(self, field_coordinates):
        field_coordinates = np.asarray(field_coordinates, dtype=float)
        if field_coordinates.shape != (self.scalar_modes,):
            raise ValueError("field_coordinates must have shape (scalar_modes,).")
        derivative = -1j * self.derivative_matrix
        scalar_potential = self.fermion_mass * np.eye(self.fermion_modes)
        for coord, vertex in zip(field_coordinates, self.scalar_vertices):
            scalar_potential = scalar_potential + self.coupling * coord * vertex
        h = np.kron(SIGMA_Z, derivative) + np.kron(SIGMA_X, scalar_potential)
        return 0.5 * (h + h.T.conj())

    def fermion_one_body_vertices(self):
        """One-body matrices ``d h_F / d q_a`` for scalar field modes."""
        return [self.coupling * np.kron(SIGMA_X, vertex) for vertex in self.scalar_vertices]

    def conditional_occupied_orbitals(self, field_coordinates):
        """Occupied one-body orbitals of the conditional fermion vacuum."""
        _, orbitals = np.linalg.eigh(self.fermion_one_body_hamiltonian(field_coordinates))
        return orbitals[:, : self.noccupied]

    def fermion_vacuum_overlap(self, bra_field_coordinates, ket_field_coordinates):
        """Slater determinant overlap ``<Omega_F[bra]|Omega_F[ket]>``."""
        bra = self.conditional_occupied_orbitals(bra_field_coordinates)
        ket = self.conditional_occupied_orbitals(ket_field_coordinates)
        return _small_determinant(bra.conj().T @ ket)

    def fermion_vacuum_overlap_matrix(self, bra_field_coordinates, ket_field_coordinates=None):
        """Pairwise conditional-vacuum overlaps for two sets of fields."""
        bra_fields = np.asarray(bra_field_coordinates, dtype=float)
        if bra_fields.ndim == 1:
            bra_fields = bra_fields.reshape(1, -1)
        ket_fields = bra_fields if ket_field_coordinates is None else np.asarray(ket_field_coordinates, dtype=float)
        if ket_fields.ndim == 1:
            ket_fields = ket_fields.reshape(1, -1)
        if bra_fields.shape[1] != self.scalar_modes or ket_fields.shape[1] != self.scalar_modes:
            raise ValueError("field coordinates must have shape (nfields, scalar_modes).")

        bra_orbitals = [self.conditional_occupied_orbitals(coords) for coords in bra_fields]
        ket_orbitals = [self.conditional_occupied_orbitals(coords) for coords in ket_fields]
        out = np.empty((len(bra_orbitals), len(ket_orbitals)), dtype=complex)
        for i, bra in enumerate(bra_orbitals):
            for j, ket in enumerate(ket_orbitals):
                out[i, j] = _small_determinant(bra.conj().T @ ket)
        return out

    def fermion_overlap_metric(self, field_coordinates, step: float = 1e-3):
        """Quantum metric from finite overlaps of conditional vacua."""
        center = np.asarray(field_coordinates, dtype=float)
        if center.shape != (self.scalar_modes,):
            raise ValueError("field_coordinates must have shape (scalar_modes,).")
        step = float(step)
        if step <= 0:
            raise ValueError("step must be positive.")

        def overlap_distance(displacement):
            shifted = center + np.asarray(displacement, dtype=float)
            amplitude = abs(self.fermion_vacuum_overlap(center, shifted))
            amplitude = min(1.0, max(float(amplitude), np.finfo(float).tiny))
            if 1.0 - amplitude < 100.0 * np.finfo(float).eps:
                return 0.0
            return -2.0 * np.log(amplitude)

        metric = np.zeros((self.scalar_modes, self.scalar_modes), dtype=float)
        unit = np.eye(self.scalar_modes)
        diagonal_distance = np.zeros(self.scalar_modes, dtype=float)
        for a in range(self.scalar_modes):
            diagonal_distance[a] = overlap_distance(step * unit[a])
            metric[a, a] = diagonal_distance[a] / (step * step)
        for a in range(self.scalar_modes):
            for b in range(a + 1, self.scalar_modes):
                combined = overlap_distance(step * (unit[a] + unit[b]))
                metric[a, b] = 0.5 * (combined / (step * step) - metric[a, a] - metric[b, b])
                metric[b, a] = metric[a, b]
        return 0.5 * (metric + metric.T)

    def fermion_vacuum_response(self, field_coordinates):
        """Energy gradient, Hessian, and quantum metric of the Dirac sea.

        This is ordinary Rayleigh-Schrödinger response of the filled
        conditional one-body levels.  It avoids sampling over field
        configurations: the only fermion solve is at ``field_coordinates``.
        """
        h = self.fermion_one_body_hamiltonian(field_coordinates)
        energies, orbitals = np.linalg.eigh(h)
        occ = np.arange(self.noccupied)
        virt = np.arange(self.noccupied, self.nspin_orbitals)
        vertices = self.fermion_one_body_vertices()
        transformed = [orbitals.conj().T @ vertex @ orbitals for vertex in vertices]

        vacuum_energy = float(np.real(np.sum(energies[occ])))
        gradient = np.zeros(self.scalar_modes, dtype=float)
        hessian = np.zeros((self.scalar_modes, self.scalar_modes), dtype=float)
        metric = np.zeros((self.scalar_modes, self.scalar_modes), dtype=float)

        for a, vertex_a in enumerate(transformed):
            gradient[a] = float(np.real(np.trace(vertex_a[np.ix_(occ, occ)])))
            for b, vertex_b in enumerate(transformed):
                hess = 0.0 + 0.0j
                qmetric = 0.0 + 0.0j
                for i in occ:
                    for r in virt:
                        denom = energies[i] - energies[r]
                        if abs(denom) == 0:
                            continue
                        matrix_product = vertex_a[i, r] * vertex_b[r, i]
                        hess += 2.0 * matrix_product / denom
                        qmetric += matrix_product / (denom * denom)
                hessian[a, b] = float(np.real(hess))
                metric[a, b] = float(np.real(qmetric))
        hessian = 0.5 * (hessian + hessian.T)
        metric = 0.5 * (metric + metric.T)
        return vacuum_energy, gradient, hessian, metric

    def conditional_fermion_gaussians(self, q_samples):
        q_samples = np.asarray(q_samples, dtype=float)
        vectors = np.empty((q_samples.shape[0], len(self.fermion_sector_basis)), dtype=complex)
        for idx, coords in enumerate(q_samples):
            _, orbitals = np.linalg.eigh(self.fermion_one_body_hamiltonian(coords))
            occupied = orbitals[:, : self.noccupied]
            vectors[idx] = slater_sector_vector(occupied, self.fermion_sector_basis)
            if idx == 0:
                pivot = int(np.argmax(np.abs(vectors[idx])))
                if abs(vectors[idx, pivot]) > 0:
                    vectors[idx] *= np.exp(-1j * np.angle(vectors[idx, pivot]))
            else:
                overlap = np.vdot(vectors[idx - 1], vectors[idx])
                if abs(overlap) > 0:
                    vectors[idx] *= np.exp(-1j * np.angle(overlap))
        return vectors

    def conditional_wavefunction(self, normalize: bool = True, widths=None, centers=None):
        q_samples, weights = self.field_quadrature()
        values = self.boson_reference(q_samples, widths=widths, centers=centers)[:, None]
        values = values * self.conditional_fermion_gaussians(q_samples)
        if normalize:
            norm = np.sqrt(np.sum(weights * np.sum(np.abs(values) ** 2, axis=1)))
            if norm == 0:
                raise ValueError("conditional wavefunction has zero norm.")
            values = values / norm
        return q_samples, weights, values

    def schmidt_compress(self, rank: int):
        rank = int(rank)
        if rank < 1:
            raise ValueError("rank must be positive.")
        q_samples, weights, values = self.conditional_wavefunction(normalize=True)
        weighted = np.sqrt(weights)[:, None] * values
        left, singular_values, right_h = np.linalg.svd(weighted, full_matrices=False)
        rank = min(rank, singular_values.size)
        weighted_compressed = (left[:, :rank] * singular_values[:rank]) @ right_h[:rank]
        compressed_values = weighted_compressed / np.sqrt(weights)[:, None]
        branches = left[:, :rank] / np.sqrt(weights)[:, None]
        coefficients, boson_basis = self.oscillator_coefficients(compressed_values, q_samples, weights)
        energy, coeff_norm = self.energy_from_coefficients(coefficients)
        exact = self.exact_ground_energy()
        return Yukawa1DNARGResult(
            rank=rank,
            field_coordinates=q_samples,
            weights=weights,
            singular_values=singular_values,
            boson_branches=branches,
            fermion_branches=right_h[:rank].copy(),
            wavefunction_values=compressed_values,
            oscillator_coefficients=coefficients,
            energy=float(energy),
            exact_energy=float(exact),
            discarded_weight=float(np.sum(singular_values[rank:] ** 2)),
            coefficient_norm=float(coeff_norm),
            fermion_basis=self.fermion_sector_basis,
            boson_basis=boson_basis,
        )

    def rank1_energy(self, widths=None, centers=None):
        """Energy of ``chi[phi] |Omega_F[phi]>`` for one Gaussian branch."""
        q_samples, weights, values = self.conditional_wavefunction(
            normalize=True,
            widths=widths,
            centers=centers,
        )
        coefficients, _ = self.oscillator_coefficients(values, q_samples, weights)
        energy, coeff_norm = self.energy_from_coefficients(coefficients)
        return energy, coeff_norm, coefficients

    def heat_kernel_kinetic_weights(self, cutoff=np.inf):
        """Mode weights from regulating the kinetic operator, not ``S``.

        The continuum smoothing is ``D_Lambda = exp(Delta / 2 Lambda^2) D``.
        For sine modes this gives kinetic weights
        ``exp(-k_a^2 / Lambda^2)`` in ``T_Lambda``.
        """
        if cutoff is None or np.isinf(cutoff):
            return np.ones(self.scalar_modes, dtype=float)
        cutoff = float(cutoff)
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        modes = np.arange(1, self.scalar_modes + 1, dtype=float)
        wave_numbers = np.pi * modes / self.length
        return np.exp(-(wave_numbers * wave_numbers) / (cutoff * cutoff))

    def ts_regulated_rank1_energy(
        self,
        widths=None,
        centers=None,
        *,
        cutoff=np.inf,
        shift: float = 1e-3,
    ):
        """Gaussian ``chi`` energy with the heat-kernel regulated ``T S`` product.

        This keeps the conditional overlap ``S[q_+, q_-]`` unregularized and
        uses the cutoff only on the finite-difference directions entering the
        scalar kinetic operator.
        """
        widths = self.scalar_frequencies if widths is None else np.asarray(widths, dtype=float)
        centers = np.zeros(self.scalar_modes) if centers is None else np.asarray(centers, dtype=float)
        if widths.shape != (self.scalar_modes,) or centers.shape != (self.scalar_modes,):
            raise ValueError("widths and centers must have shape (scalar_modes,).")
        if np.any(widths <= 0):
            raise ValueError("all Gaussian widths must be positive.")
        shift = float(shift)
        if shift <= 0:
            raise ValueError("shift must be positive.")

        q_samples, weights = self.field_quadrature()
        chi = self.boson_reference(q_samples, widths=widths, centers=centers)
        density = np.abs(chi) ** 2
        norm = float(np.sum(weights * density))
        if norm <= 0:
            raise ValueError("chi has zero quadrature norm.")

        omega2 = self.scalar_frequencies * self.scalar_frequencies
        boson_potential_values = 0.5 * np.sum(omega2 * q_samples * q_samples, axis=1)
        fermion_values = np.empty(q_samples.shape[0], dtype=float)
        for sample_index, coords in enumerate(q_samples):
            levels = np.linalg.eigvalsh(self.fermion_one_body_hamiltonian(coords))
            fermion_values[sample_index] = float(np.sum(levels[: self.noccupied]))

        kinetic_weights = self.heat_kernel_kinetic_weights(cutoff)
        kinetic_values = np.zeros(q_samples.shape[0], dtype=float)
        directions = np.eye(self.scalar_modes)
        for mode, direction in enumerate(directions):
            mode_values = np.empty(q_samples.shape[0], dtype=float)
            for sample_index, coords in enumerate(q_samples):
                q_plus = coords + 0.5 * shift * direction
                q_minus = coords - 0.5 * shift * direction
                chi_plus = self.boson_reference(q_plus[None, :], widths=widths, centers=centers)[0]
                chi_minus = self.boson_reference(q_minus[None, :], widths=widths, centers=centers)[0]
                # Local kinetic differences use the parallel-transport gauge.
                overlap = abs(self.fermion_vacuum_overlap(q_plus, q_minus))
                distance = (
                    abs(chi_plus) ** 2
                    + abs(chi_minus) ** 2
                    - 2.0 * np.real(np.conj(chi_plus) * chi_minus) * overlap
                )
                mode_values[sample_index] = max(0.0, float(distance)) / (shift * shift)
            kinetic_values += 0.5 * kinetic_weights[mode] * mode_values

        kinetic_energy = float(np.sum(weights * kinetic_values) / norm)
        boson_potential_energy = float(np.sum(weights * density * boson_potential_values) / norm)
        fermion_energy = float(np.sum(weights * density * fermion_values) / norm)
        energy = kinetic_energy + boson_potential_energy + fermion_energy
        return Yukawa1DRegulatedKineticRank1Result(
            widths=widths.copy(),
            centers=centers.copy(),
            cutoff=float(cutoff) if cutoff is not None else np.inf,
            shift=shift,
            kinetic_weights=kinetic_weights.copy(),
            energy=float(energy),
            exact_energy=self.exact_ground_energy(),
            kinetic_energy=kinetic_energy,
            boson_potential_energy=boson_potential_energy,
            fermion_energy=fermion_energy,
            norm=norm,
        )

    def boson_gaussian_energy(self, widths=None, centers=None):
        """Analytic boson energy for a real diagonal Gaussian wavefunctional."""
        widths = self.scalar_frequencies if widths is None else np.asarray(widths, dtype=float)
        centers = np.zeros(self.scalar_modes) if centers is None else np.asarray(centers, dtype=float)
        if widths.shape != (self.scalar_modes,) or centers.shape != (self.scalar_modes,):
            raise ValueError("widths and centers must have shape (scalar_modes,).")
        if np.any(widths <= 0):
            raise ValueError("all Gaussian widths must be positive.")
        omega2 = self.scalar_frequencies * self.scalar_frequencies
        kinetic = 0.25 * np.sum(widths)
        potential_mean = 0.5 * np.sum(omega2 * centers * centers)
        potential_width = 0.25 * np.sum(omega2 / widths)
        return float(kinetic + potential_mean + potential_width)

    def gaussian_response_rank1_energy(
        self,
        widths=None,
        centers=None,
        *,
        include_born_huang: bool = True,
        metric_source: str = "overlap",
        overlap_step: float = 1e-3,
    ):
        """Analytic D=1 Gaussian-response energy, no field sampling.

        The fermion vacuum energy is expanded to second order around the
        Gaussian center.  The Gaussian average then contributes
        ``1/4 Tr(Hessian / widths)`` for diagonal widths.  The optional
        Born-Huang term uses the conditional Slater determinant quantum metric.
        """
        widths = self.scalar_frequencies if widths is None else np.asarray(widths, dtype=float)
        centers = np.zeros(self.scalar_modes) if centers is None else np.asarray(centers, dtype=float)
        if widths.shape != (self.scalar_modes,) or centers.shape != (self.scalar_modes,):
            raise ValueError("widths and centers must have shape (scalar_modes,).")
        if np.any(widths <= 0):
            raise ValueError("all Gaussian widths must be positive.")

        boson_energy = self.boson_gaussian_energy(widths=widths, centers=centers)
        fermion_energy, gradient, hessian, response_metric = self.fermion_vacuum_response(centers)
        metric_source = str(metric_source).lower().replace("-", "_")
        if metric_source == "overlap":
            metric = self.fermion_overlap_metric(centers, step=overlap_step)
        elif metric_source in {"response", "born_huang", "derivative"}:
            metric = response_metric
        elif metric_source in {"none", "off"}:
            metric = np.zeros_like(response_metric)
        else:
            raise ValueError("metric_source must be 'overlap', 'response', or 'none'.")
        covariance_diag = 0.5 / widths
        fluctuation_energy = 0.5 * float(np.sum(np.diag(hessian) * covariance_diag))
        born_huang_energy = 0.5 * float(np.trace(metric)) if include_born_huang else 0.0
        energy = boson_energy + fermion_energy + fluctuation_energy + born_huang_energy
        return Yukawa1DGaussianResponseResult(
            widths=widths.copy(),
            centers=centers.copy(),
            energy=float(energy),
            exact_energy=self.exact_ground_energy(),
            boson_energy=float(boson_energy),
            fermion_energy=float(fermion_energy),
            fluctuation_energy=float(fluctuation_energy),
            born_huang_energy=float(born_huang_energy),
            fermion_gradient=gradient.copy(),
            fermion_hessian=hessian.copy(),
            quantum_metric=metric.copy(),
            metric_source=metric_source,
        )

    def variational_rank1_response(
        self,
        *,
        initial_widths=None,
        initial_centers=None,
        optimize_widths: bool = True,
        optimize_centers: bool = True,
        include_born_huang: bool = True,
        metric_source: str = "overlap",
        overlap_step: float = 1e-3,
        maxiter: int = 200,
        method: str = "Powell",
    ):
        """Optimize the analytic quadratic-response D=1 Gaussian energy."""
        from scipy.optimize import minimize

        widths0 = (
            self.scalar_frequencies.copy()
            if initial_widths is None
            else np.asarray(initial_widths, dtype=float).copy()
        )
        centers0 = (
            np.zeros(self.scalar_modes)
            if initial_centers is None
            else np.asarray(initial_centers, dtype=float).copy()
        )
        if widths0.shape != (self.scalar_modes,) or centers0.shape != (self.scalar_modes,):
            raise ValueError("initial_widths and initial_centers must have shape (scalar_modes,).")
        if np.any(widths0 <= 0):
            raise ValueError("initial_widths must be positive.")

        parts = []
        x0 = []
        if optimize_widths:
            parts.append(("log_widths", self.scalar_modes))
            x0.extend(np.log(widths0))
        if optimize_centers:
            parts.append(("centers", self.scalar_modes))
            x0.extend(centers0)
        x0 = np.asarray(x0, dtype=float)

        def unpack(params):
            params = np.asarray(params, dtype=float)
            widths = widths0.copy()
            centers = centers0.copy()
            cursor = 0
            for name, size in parts:
                block = params[cursor : cursor + size]
                cursor += size
                if name == "log_widths":
                    widths = np.exp(block)
                elif name == "centers":
                    centers = block.copy()
            return widths, centers

        def objective(params):
            widths, centers = unpack(params)
            return self.gaussian_response_rank1_energy(
                widths=widths,
                centers=centers,
                include_born_huang=include_born_huang,
                metric_source=metric_source,
                overlap_step=overlap_step,
            ).energy

        if parts:
            result = minimize(
                objective,
                x0,
                method=method,
                options={"maxiter": int(maxiter)},
            )
            widths, centers = unpack(result.x)
            out = self.gaussian_response_rank1_energy(
                widths=widths,
                centers=centers,
                include_born_huang=include_born_huang,
                metric_source=metric_source,
                overlap_step=overlap_step,
            )
            out.success = bool(result.success)
            out.message = str(result.message)
            out.nfev = int(result.nfev)
            return out

        out = self.gaussian_response_rank1_energy(
            widths=widths0,
            centers=centers0,
            include_born_huang=include_born_huang,
            metric_source=metric_source,
            overlap_step=overlap_step,
        )
        out.message = "no variational parameters"
        return out

    def _normalize_packet_parameters(self, widths, centers):
        centers = np.asarray(centers, dtype=float)
        if centers.ndim == 1:
            centers = centers.reshape(1, -1)
        if centers.ndim != 2 or centers.shape[1] != self.scalar_modes:
            raise ValueError("centers must have shape (npackets, scalar_modes).")

        widths = np.asarray(widths, dtype=float)
        if widths.ndim == 0:
            widths = np.full((centers.shape[0], self.scalar_modes), float(widths))
        elif widths.ndim == 1:
            if widths.shape != (self.scalar_modes,):
                raise ValueError("widths must have shape (scalar_modes,) or (npackets, scalar_modes).")
            widths = np.broadcast_to(widths, centers.shape).copy()
        elif widths.ndim == 2:
            if widths.shape != centers.shape:
                raise ValueError("widths must have shape (scalar_modes,) or (npackets, scalar_modes).")
        else:
            raise ValueError("widths must have shape (scalar_modes,) or (npackets, scalar_modes).")
        if np.any(widths <= 0):
            raise ValueError("all Gaussian packet widths must be positive.")
        return widths.copy(), centers.copy()

    @staticmethod
    def _gaussian_pair_moments(width_i, center_i, width_j, center_j):
        width_i = float(width_i)
        width_j = float(width_j)
        center_i = float(center_i)
        center_j = float(center_j)
        denom = width_i + width_j
        delta = center_j - center_i
        overlap = np.sqrt(2.0 * np.sqrt(width_i * width_j) / denom) * np.exp(
            -0.5 * width_i * width_j * delta * delta / denom
        )
        mean = (width_i * center_i + width_j * center_j) / denom
        variance = 1.0 / denom
        q = overlap * mean
        q2 = overlap * (mean * mean + variance)
        kinetic = (
            0.5
            * width_i
            * width_j
            * overlap
            * (variance - width_i * width_j * delta * delta / (denom * denom))
        )
        return float(overlap), float(q), float(q2), float(kinetic)

    def gaussian_packet_boson_matrices(self, widths, centers):
        """Bosonic overlap, Hamiltonian, and ``q_a`` matrices for packets."""
        widths, centers = self._normalize_packet_parameters(widths, centers)
        npackets = centers.shape[0]
        mode_overlap = np.empty((self.scalar_modes, npackets, npackets), dtype=float)
        mode_q = np.empty_like(mode_overlap)
        mode_q2 = np.empty_like(mode_overlap)
        mode_kinetic = np.empty_like(mode_overlap)

        for mode in range(self.scalar_modes):
            for i in range(npackets):
                for j in range(npackets):
                    overlap, q, q2, kinetic = self._gaussian_pair_moments(
                        widths[i, mode],
                        centers[i, mode],
                        widths[j, mode],
                        centers[j, mode],
                    )
                    mode_overlap[mode, i, j] = overlap
                    mode_q[mode, i, j] = q
                    mode_q2[mode, i, j] = q2
                    mode_kinetic[mode, i, j] = kinetic

        boson_overlap = np.prod(mode_overlap, axis=0)
        boson_hamiltonian = np.zeros((npackets, npackets), dtype=float)
        q_matrices = []
        for mode, omega in enumerate(self.scalar_frequencies):
            other_overlap = np.prod(np.delete(mode_overlap, mode, axis=0), axis=0)
            boson_hamiltonian += other_overlap * (
                mode_kinetic[mode] + 0.5 * omega * omega * mode_q2[mode]
            )
            q_matrices.append(other_overlap * mode_q[mode])

        boson_overlap = 0.5 * (boson_overlap + boson_overlap.T)
        boson_hamiltonian = 0.5 * (boson_hamiltonian + boson_hamiltonian.T)
        q_matrices = [0.5 * (q_matrix + q_matrix.T) for q_matrix in q_matrices]
        return boson_overlap, boson_hamiltonian, q_matrices

    def gaussian_packet_matrices(self, widths, centers, *, return_parts: bool = False):
        """Projected matrices for ``chi_i[q] |Omega_F(q_i)>`` packets.

        The fermion overlap matrix multiplies every purely bosonic matrix
        element, so the conditional Gram kernel dresses the scalar kinetic
        operator before the generalized eigenproblem is solved.
        """
        widths, centers = self._normalize_packet_parameters(widths, centers)
        boson_overlap, boson_hamiltonian, q_matrices = self.gaussian_packet_boson_matrices(
            widths,
            centers,
        )

        fock_vectors = np.asarray(
            [
                slater_sector_vector(
                    self.conditional_occupied_orbitals(center),
                    self.fermion_sector_basis,
                )
                for center in centers
            ],
            dtype=complex,
        )
        fermion_overlap = fock_vectors.conj() @ fock_vectors.T

        h0 = self.fermion_one_body_hamiltonian(np.zeros(self.scalar_modes))
        fermion_h0_sector = one_body_sector_matrix(h0, self.noccupied)
        fermion_h0 = fock_vectors.conj() @ fermion_h0_sector @ fock_vectors.T

        fermion_vertices = []
        for vertex in self.fermion_one_body_vertices():
            vertex_sector = one_body_sector_matrix(vertex, self.noccupied)
            fermion_vertices.append(fock_vectors.conj() @ vertex_sector @ fock_vectors.T)

        boson_dressed = boson_hamiltonian * fermion_overlap
        hamiltonian = boson_dressed + boson_overlap * fermion_h0
        for q_matrix, vertex_matrix in zip(q_matrices, fermion_vertices):
            hamiltonian = hamiltonian + q_matrix * vertex_matrix
        overlap = boson_overlap * fermion_overlap

        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T.conj())
        overlap = 0.5 * (overlap + overlap.T.conj())

        parts = {
            "boson_overlap": boson_overlap,
            "boson_hamiltonian": boson_hamiltonian,
            "boson_dressed": boson_dressed,
            "boson_q": q_matrices,
            "fermion_overlap": fermion_overlap,
            "fermion_h0": fermion_h0,
            "fermion_vertices": fermion_vertices,
        }
        if return_parts:
            return hamiltonian, overlap, parts
        return hamiltonian, overlap

    def gaussian_packet_ground_state(self, widths, centers):
        """Solve the nonorthogonal Gaussian-packet NARG backend."""
        from scipy.linalg import eigh

        widths, centers = self._normalize_packet_parameters(widths, centers)
        hamiltonian, overlap, parts = self.gaussian_packet_matrices(widths, centers, return_parts=True)
        values, vectors = eigh(hamiltonian, overlap)
        coefficients = vectors[:, 0]
        norm = np.vdot(coefficients, overlap @ coefficients)
        coefficients = coefficients / np.sqrt(norm)
        return Yukawa1DGaussianPacketResult(
            widths=widths,
            centers=centers,
            energy=float(np.real(values[0])),
            exact_energy=self.exact_ground_energy(),
            coefficients=coefficients,
            hamiltonian=hamiltonian,
            overlap=overlap,
            parts=parts,
        )

    def variational_rank1(
        self,
        *,
        initial_widths=None,
        initial_centers=None,
        optimize_widths: bool = True,
        optimize_centers: bool = True,
        maxiter: int = 200,
        method: str = "Powell",
    ):
        """Optimize a single conditional branch over Gaussian ``chi[phi]``.

        The variational ansatz is

        ``Psi[phi] = chi_{width, center}[phi] |Omega_F[phi]>``.

        This is a direct wavefunction variational calculation with ``D=1``:
        no effective action and no multi-branch Schmidt expansion.
        """
        from scipy.optimize import minimize

        widths0 = (
            self.scalar_frequencies.copy()
            if initial_widths is None
            else np.asarray(initial_widths, dtype=float).copy()
        )
        centers0 = (
            np.zeros(self.scalar_modes)
            if initial_centers is None
            else np.asarray(initial_centers, dtype=float).copy()
        )
        if widths0.shape != (self.scalar_modes,) or centers0.shape != (self.scalar_modes,):
            raise ValueError("initial_widths and initial_centers must have shape (scalar_modes,).")
        if np.any(widths0 <= 0):
            raise ValueError("initial_widths must be positive.")

        parts = []
        x0 = []
        if optimize_widths:
            parts.append(("log_widths", self.scalar_modes))
            x0.extend(np.log(widths0))
        if optimize_centers:
            parts.append(("centers", self.scalar_modes))
            x0.extend(centers0)
        x0 = np.asarray(x0, dtype=float)

        def unpack(params):
            params = np.asarray(params, dtype=float)
            widths = widths0.copy()
            centers = centers0.copy()
            cursor = 0
            for name, size in parts:
                block = params[cursor : cursor + size]
                cursor += size
                if name == "log_widths":
                    widths = np.exp(block)
                elif name == "centers":
                    centers = block.copy()
            return widths, centers

        def objective(params):
            widths, centers = unpack(params)
            energy, _, _ = self.rank1_energy(widths=widths, centers=centers)
            return energy

        if not parts:
            energy, coeff_norm, coeff = self.rank1_energy(widths=widths0, centers=centers0)
            return Yukawa1DVariationalRank1Result(
                widths=widths0,
                centers=centers0,
                energy=float(energy),
                exact_energy=self.exact_ground_energy(),
                coefficient_norm=float(coeff_norm),
                oscillator_coefficients=coeff,
                success=True,
                message="no variational parameters",
                nfev=1,
            )

        result = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": int(maxiter)},
        )
        widths, centers = unpack(result.x)
        energy, coeff_norm, coeff = self.rank1_energy(widths=widths, centers=centers)
        return Yukawa1DVariationalRank1Result(
            widths=widths,
            centers=centers,
            energy=float(energy),
            exact_energy=self.exact_ground_energy(),
            coefficient_norm=float(coeff_norm),
            oscillator_coefficients=coeff,
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
        )

    def oscillator_coefficients(self, wavefunction_values, q_samples=None, weights=None):
        if q_samples is None or weights is None:
            q_samples, weights = self.field_quadrature()
        basis_values, boson_basis = product_oscillator_basis_values(
            q_samples, self.oscillator_nbasis, self.scalar_frequencies
        )
        coefficients = basis_values.T @ (weights[:, None] * np.asarray(wavefunction_values))
        return coefficients, boson_basis

    def boson_hamiltonian_matrix(self):
        dim = self.oscillator_nbasis
        identity = np.eye(dim)
        total_dim = dim ** self.scalar_modes
        hamiltonian = np.zeros((total_dim, total_dim), dtype=complex)
        for mode, omega in enumerate(self.scalar_frequencies):
            _, _, h_mode = oscillator_operators(dim, omega)
            ops = [identity] * self.scalar_modes
            ops[mode] = h_mode
            hamiltonian += _kron_all(ops)
        return 0.5 * (hamiltonian + hamiltonian.T.conj())

    def boson_q_operator(self, mode: int):
        dim = self.oscillator_nbasis
        identity = np.eye(dim)
        q_mode, _, _ = oscillator_operators(dim, self.scalar_frequencies[int(mode)])
        ops = [identity] * self.scalar_modes
        ops[int(mode)] = q_mode
        return _kron_all(ops)

    def hamiltonian_matrix(self):
        boson_h = self.boson_hamiltonian_matrix()
        boson_dim = boson_h.shape[0]
        fock_dim = len(self.fermion_sector_basis)
        h0 = self.fermion_one_body_hamiltonian(np.zeros(self.scalar_modes))
        fermion_h = one_body_sector_matrix(h0, self.noccupied)
        hamiltonian = np.kron(boson_h, np.eye(fock_dim)) + np.kron(np.eye(boson_dim), fermion_h)
        for mode, vertex in enumerate(self.scalar_vertices):
            yukawa_one_body = self.coupling * np.kron(SIGMA_X, vertex)
            yukawa_sector = one_body_sector_matrix(yukawa_one_body, self.noccupied)
            hamiltonian += np.kron(self.boson_q_operator(mode), yukawa_sector)
        return 0.5 * (hamiltonian + hamiltonian.T.conj())

    def energy_from_coefficients(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=complex)
        vector = coefficients.reshape(-1)
        norm = np.vdot(vector, vector)
        if abs(norm) == 0:
            raise ValueError("coefficient vector has zero norm.")
        hamiltonian = self.hamiltonian_matrix()
        energy = np.vdot(vector, hamiltonian @ vector) / norm
        return float(np.real(energy)), float(np.real(norm))

    def exact_ground_energy(self):
        if self._exact_ground_energy_cache is None:
            self._exact_ground_energy_cache = float(np.linalg.eigvalsh(self.hamiltonian_matrix())[0])
        return float(self._exact_ground_energy_cache)


__all__ = [
    "ConditionalGaussianNARGResult",
    "ConditionalGaussianWavefunctionNARG",
    "Phi4LogShellCouplingFlowResult",
    "Phi4LogShellCouplingFlowStep",
    "Phi4LogShellCoarseGrainResult",
    "Phi4LogShellIterativeNARGResult",
    "Phi4LogShellNARG",
    "Phi4LogShellNARGResult",
    "Phi4MomentumSpaceNARG",
    "Phi4MomentumSpaceNARGResult",
    "Phi4MomentumSpaceNARGStepResult",
    "Phi4NARGEffectiveHamiltonianResult",
    "Phi4PeriodicSincNARG",
    "Phi4PeriodicSincNARGResult",
    "Phi4TwoSiteNARG",
    "Yukawa1DGaussianPacketResult",
    "Yukawa1DGaussianResponseResult",
    "Yukawa1DNARGResult",
    "Yukawa1DRegulatedKineticRank1Result",
    "Yukawa1DVariationalRank1Result",
    "Yukawa1DWavefunctionalNARG",
    "fixed_particle_basis",
    "hermite_function_values",
    "hermite_quadrature",
    "interval_legendre_quadrature",
    "oscillator_operators",
    "one_body_sector_matrix",
    "periodic_real_fourier_transform",
    "periodic_sinc_grid",
    "sine_basis_derivative_matrix",
    "sine_basis_derivative_values",
    "sine_basis_values",
    "sine_dvr_derivative_matrix",
    "sine_dvr_grid",
    "sine_dvr_kinetic_matrix",
    "sine_dvr_transform",
    "slater_sector_vector",
]
