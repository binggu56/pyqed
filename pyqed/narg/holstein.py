#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small exact Holstein-dimer tools for NARG/LETTA diagnostics.

The model is the one-electron, two-site Holstein dimer

    H = -t (|1><2| + |2><1|)
        + omega (b1^dag b1 + b2^dag b2)
        + g [n1 (b1^dag + b1) + n2 (b2^dag + b2)].

The electronic Hilbert space is the one-electron subspace spanned by
``|1>`` and ``|2>``.  The phonon Hilbert space is a product Fock basis with
``nphonon`` states for each mode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from math import comb
import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from .core import SequentialNARGState


def boson_annihilation(n: int, *, dtype=float) -> np.ndarray:
    """Return the truncated harmonic-oscillator annihilation operator."""
    if int(n) < 1:
        raise ValueError("n must be a positive integer.")
    op = np.zeros((int(n), int(n)), dtype=dtype)
    for level in range(1, int(n)):
        op[level - 1, level] = np.sqrt(level)
    return op


def _normalized_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("state must have nonzero norm.")
    return state / norm


def schmidt_spectrum(state: np.ndarray, left_dim: int, right_dim: int) -> np.ndarray:
    """Return Schmidt singular values for a vector reshaped as left x right."""
    state = _normalized_state(state)
    if state.size != int(left_dim) * int(right_dim):
        raise ValueError("state size is incompatible with left_dim * right_dim.")
    return np.linalg.svd(state.reshape(int(left_dim), int(right_dim)), compute_uv=False)


def discarded_weight(singular_values: np.ndarray, rank: int) -> float:
    """Return the Schmidt weight discarded after keeping ``rank`` values."""
    singular_values = np.asarray(singular_values, dtype=float)
    rank = max(0, int(rank))
    return float(np.sum(singular_values[rank:] ** 2))


def truncate_schmidt_state(
    state: np.ndarray,
    left_dim: int,
    right_dim: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized best rank-D approximation and singular values."""
    state = _normalized_state(state)
    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be at least 1.")
    if state.size != int(left_dim) * int(right_dim):
        raise ValueError("state size is incompatible with left_dim * right_dim.")

    matrix = state.reshape(int(left_dim), int(right_dim))
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, singular_values.size)
    truncated = (u[:, :keep] * singular_values[:keep]) @ vh[:keep]
    vector = truncated.reshape(-1)
    vector /= np.linalg.norm(vector)
    return vector, singular_values


def conditional_rank1_factor(
    state: np.ndarray,
    left_dim: int,
    right_dim: int,
    *,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor one vector as ``psi[i,n] = A[i,n,0] B[0,n]``.

    This is the two-site conditional LETTA/NARG form discussed as
    ``A_{i n a} B_{a n}``, specialized to a single retained channel
    ``a = 0``.  For one target vector it is exact whenever the local
    conditional vector at phonon configuration ``n`` has nonzero norm.
    """
    state = _normalized_state(state)
    if state.size != int(left_dim) * int(right_dim):
        raise ValueError("state size is incompatible with left_dim * right_dim.")

    matrix = state.reshape(int(left_dim), int(right_dim))
    weights = np.linalg.norm(matrix, axis=0)
    dtype = np.result_type(matrix.dtype, complex)
    a_tensor = np.zeros((int(left_dim), int(right_dim), 1), dtype=dtype)
    b_tensor = np.zeros((1, int(right_dim)), dtype=dtype)
    active = weights > float(tol)
    a_tensor[:, active, 0] = matrix[:, active] / weights[active]
    b_tensor[0, active] = weights[active]
    return a_tensor, b_tensor


def reconstruct_conditional_factor(a_tensor: np.ndarray, b_tensor: np.ndarray) -> np.ndarray:
    """Reconstruct ``psi[i,n]`` from ``A[i,n,a]`` and ``B[a,n]``."""
    a_tensor = np.asarray(a_tensor)
    b_tensor = np.asarray(b_tensor)
    if a_tensor.ndim != 3 or b_tensor.ndim != 2:
        raise ValueError("expected A with ndim=3 and B with ndim=2.")
    if a_tensor.shape[2] != b_tensor.shape[0] or a_tensor.shape[1] != b_tensor.shape[1]:
        raise ValueError("A and B have incompatible channel/physical dimensions.")
    return np.einsum("ina,an->in", a_tensor, b_tensor)


@dataclass(frozen=True)
class RankDFrameReport:
    """Rank-D diagnostics for one frame of the Holstein-dimer state."""

    eta: float | None
    singular_values: np.ndarray
    discarded_weights: dict[int, float]
    energies: dict[int, float]


@dataclass(frozen=True)
class HolsteinDimerReport:
    """Exact and rank-D diagnostics for the bare and disentangled frames."""

    model: "HolsteinDimer"
    exact_energies: np.ndarray
    bare: RankDFrameReport
    lang_firsov: RankDFrameReport


@dataclass(frozen=True)
class HolsteinDimerConditionalResult:
    """Result of a coordinate-grid conditional-state Holstein calculation."""

    energies: np.ndarray
    vectors: np.ndarray
    conditional_energies: np.ndarray
    conditional_vectors: np.ndarray
    hamiltonian: np.ndarray
    nstates_per_point: int


@dataclass(frozen=True)
class HolsteinChainBlock:
    """Compressed Holstein-chain block in zero- and one-electron sectors."""

    h0: np.ndarray
    h1: np.ndarray
    c_boundary: np.ndarray


@dataclass(frozen=True)
class HolsteinDressedSite:
    """Local zero-/one-electron dressed site used by Holstein-chain NARG."""

    h0: np.ndarray
    h1: np.ndarray
    c: np.ndarray


@dataclass(frozen=True)
class HolsteinChainNARGResult:
    """Result of the recursive Fock-space Holstein NARG calculation."""

    energies: np.ndarray
    block: HolsteinChainBlock
    sector_dims: list[tuple[int, int]]


@dataclass(frozen=True)
class HolsteinAdiabaticStep:
    """One site-addition step in the explicit conditional-basis NARG form."""

    site: int
    conditional_dim: int
    raw_dim: int
    states_per_branch: tuple[int, int]
    orthonormal_dim: int | None = None
    site_eigenvalues: np.ndarray | None = None
    site_annihilation_expectations: np.ndarray | None = None
    site_p_expectations: np.ndarray | None = None
    overlap_eigenvalues: np.ndarray | None = None


@dataclass(frozen=True)
class HolsteinAdiabaticNARGResult:
    """Result of explicit adiabatic/conditional Holstein NARG growth."""

    energies: np.ndarray
    block: HolsteinChainBlock
    sector_dims: list[tuple[int, int]]
    steps: list[HolsteinAdiabaticStep]


@dataclass(frozen=True)
class HolsteinElectronicFirstStep:
    """One active-mode addition in electronic-first Holstein NARG."""

    mode: int
    product_dim: int
    kept: int
    lowest_energy: float


@dataclass(frozen=True)
class HolsteinElectronicFirstResult:
    """Result of Holstein NARG with the electronic system as the first site."""

    energies: np.ndarray
    block_hamiltonian: np.ndarray
    density_operators: list[np.ndarray]
    steps: list[HolsteinElectronicFirstStep]


@dataclass(frozen=True)
class SpinfulHolsteinElectronicFirstResult:
    """Result of spinful half-filled Holstein NARG with electronic first site."""

    energies: np.ndarray
    block_hamiltonian: np.ndarray
    density_operators: list[np.ndarray]
    target: tuple[int, int]
    electronic_dim: int
    steps: list[HolsteinElectronicFirstStep]


@dataclass(frozen=True)
class SpinfulHolsteinModeTransform:
    """Collective phonon modes ordered by electronic density response."""

    transform: np.ndarray
    strengths: np.ndarray
    gram: np.ndarray
    electronic_energies: np.ndarray
    nlow: int
    centered: bool


@dataclass(frozen=True)
class SpinfulHolsteinAdiabaticElectronicResult:
    """Result of active-mode NARG with conditional electronic states."""

    energies: np.ndarray
    vectors: np.ndarray
    conditional_energies: np.ndarray
    conditional_vectors: np.ndarray
    hamiltonian: np.ndarray
    nstates_per_point: int
    active_modes: tuple[int, ...]
    target: tuple[int, int]
    mode_transform: np.ndarray | None = None
    mode_strengths: np.ndarray | None = None


@dataclass(frozen=True)
class SpinfulHolsteinSequentialAdiabaticStep:
    """One sequential active-mode addition in adiabatic electronic NARG."""

    mode: int
    input_dim: int
    grid_dim: int
    conditional_dim: int
    hamiltonian_dim: int
    kept: int
    lowest_energy: float


@dataclass(frozen=True)
class SpinfulHolsteinSequentialAdiabaticResult:
    """Result of adding active phonon modes one at a time."""

    energies: np.ndarray
    block_hamiltonian: np.ndarray
    density_operators: list[np.ndarray]
    target: tuple[int, int]
    electronic_dim: int
    steps: list[SpinfulHolsteinSequentialAdiabaticStep]
    mode_transform: np.ndarray | None = None
    mode_strengths: np.ndarray | None = None
    narg_tensors: tuple[np.ndarray, ...] | None = None
    narg_coefficients: np.ndarray | None = None
    narg_dims: tuple[int, ...] | None = None
    narg_electronic_basis: np.ndarray | None = None
    narg_electronic_hamiltonian: np.ndarray | None = None
    narg_density_operators: tuple[np.ndarray, ...] | None = None

    def narg_state(self, root: int = 0) -> SequentialNARGState:
        """Return the exported electronic-first sequential NARG state."""
        if self.narg_tensors is None or self.narg_coefficients is None:
            raise ValueError("this result was not run with store_narg_state=True.")
        return SequentialNARGState(
            list(self.narg_tensors),
            self.narg_coefficients,
            dims=self.narg_dims,
            root=int(root),
        )

    def to_letta(
        self,
        *,
        root: int = 0,
        hamiltonian=None,
        bond_dim: int | None = None,
        overlap=None,
        seed=None,
        local_masks=None,
        preserve_support: bool = False,
        support_tol: float = 1e-12,
        append_terminal: bool = False,
    ):
        """Initialize LETTA directly from the exported NARG tensors."""
        if self.narg_tensors is None or self.narg_coefficients is None:
            raise ValueError("this result was not run with store_narg_state=True.")
        from .letta import LETTA

        return LETTA.from_narg(
            list(self.narg_tensors),
            self.narg_coefficients,
            dims=self.narg_dims,
            root=int(root),
            hamiltonian=hamiltonian,
            bond_dim=bond_dim,
            overlap=overlap,
            seed=seed,
            local_masks=local_masks,
            preserve_support=preserve_support,
            support_tol=support_tol,
            append_terminal=append_terminal,
        )


@dataclass(frozen=True)
class HolsteinDimer:
    """Exact one-electron Holstein dimer with two local phonon modes."""

    t: float = 1.0
    omega: float = 1.0
    g: float = 1.0
    nphonon: int = 8

    def __post_init__(self):
        if self.nphonon < 1:
            raise ValueError("nphonon must be at least 1.")
        if self.omega == 0:
            raise ValueError("omega must be nonzero.")

    @property
    def electron_dim(self) -> int:
        return 2

    @property
    def phonon_dim(self) -> int:
        return int(self.nphonon) ** 2

    @property
    def dim(self) -> int:
        return self.electron_dim * self.phonon_dim

    def electronic_operators(self) -> dict[str, np.ndarray]:
        """Return electronic operators in the one-electron dimer basis."""
        return {
            "eye": np.eye(2),
            "n1": np.diag([1.0, 0.0]),
            "n2": np.diag([0.0, 1.0]),
            "hop": np.array([[0.0, -self.t], [-self.t, 0.0]]),
        }

    def phonon_operators(self) -> dict[str, np.ndarray]:
        """Return two-mode phonon operators in the product Fock basis."""
        n = int(self.nphonon)
        eye = np.eye(n)
        b = boson_annihilation(n)
        bdag = b.T.conj()
        num = bdag @ b
        x = b + bdag
        k = bdag - b
        return {
            "eye": np.eye(n * n),
            "num1": np.kron(num, eye),
            "num2": np.kron(eye, num),
            "x1": np.kron(x, eye),
            "x2": np.kron(eye, x),
            "k1": np.kron(k, eye),
            "k2": np.kron(eye, k),
        }

    def hamiltonian(self) -> np.ndarray:
        """Return the dense Holstein-dimer Hamiltonian."""
        eops = self.electronic_operators()
        pops = self.phonon_operators()
        return (
            np.kron(eops["hop"], pops["eye"])
            + np.kron(eops["eye"], self.omega * (pops["num1"] + pops["num2"]))
            + self.g * (np.kron(eops["n1"], pops["x1"]) + np.kron(eops["n2"], pops["x2"]))
        )

    def lang_firsov_generator(self) -> np.ndarray:
        """Return the anti-Hermitian conditional-displacement generator."""
        eops = self.electronic_operators()
        pops = self.phonon_operators()
        return np.kron(eops["n1"], pops["k1"]) + np.kron(eops["n2"], pops["k2"])

    def lang_firsov_unitary(self, eta: float | None = None) -> np.ndarray:
        """Return ``exp(eta K)`` for the conditional phonon displacement."""
        if eta is None:
            eta = -self.g / self.omega
        return expm(float(eta) * self.lang_firsov_generator())

    def transformed_hamiltonian(self, eta: float | None = None) -> np.ndarray:
        """Return ``U(eta)^dag H U(eta)``."""
        unitary = self.lang_firsov_unitary(eta)
        return unitary.conj().T @ self.hamiltonian() @ unitary

    def eigensystem(self, nstates: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return exact eigenvalues/eigenvectors of the dense Hamiltonian."""
        energies, vectors = eigh(self.hamiltonian())
        if nstates is None:
            return energies, vectors
        nstates = int(nstates)
        return energies[:nstates], vectors[:, :nstates]

    def schmidt_spectrum(self, state: np.ndarray) -> np.ndarray:
        """Return electron|phonon Schmidt values for ``state``."""
        return schmidt_spectrum(state, self.electron_dim, self.phonon_dim)

    def rank_projected_energy(
        self,
        rank: int,
        *,
        state: np.ndarray | None = None,
        eta: float | None = None,
        hamiltonian: np.ndarray | None = None,
    ) -> float:
        """Energy of the best rank-D electron|phonon state in a chosen frame.

        If ``eta`` is supplied, the exact state is first rotated to the
        Lang-Firsov frame, truncated there, and then rotated back before the
        original Hamiltonian expectation value is evaluated.
        """
        if hamiltonian is None:
            hamiltonian = self.hamiltonian()
        if state is None:
            _, vectors = self.eigensystem(nstates=1)
            state = vectors[:, 0]
        state = _normalized_state(state)

        if eta is None:
            truncated, _ = truncate_schmidt_state(
                state, self.electron_dim, self.phonon_dim, rank
            )
            trial = truncated
        else:
            unitary = self.lang_firsov_unitary(eta)
            rotated = unitary.conj().T @ state
            truncated, _ = truncate_schmidt_state(
                rotated, self.electron_dim, self.phonon_dim, rank
            )
            trial = unitary @ truncated
            trial /= np.linalg.norm(trial)

        return float(np.vdot(trial, hamiltonian @ trial).real)

    def optimize_lang_firsov_eta(
        self,
        *,
        state: np.ndarray | None = None,
        rank: int = 1,
        eta_values: np.ndarray | None = None,
        objective: str = "discarded_weight",
    ) -> tuple[float, float]:
        """Grid-search the displacement that best compresses the target state."""
        if state is None:
            _, vectors = self.eigensystem(nstates=1)
            state = vectors[:, 0]
        state = _normalized_state(state)
        if eta_values is None:
            radius = max(1.0, 2.0 * abs(self.g / self.omega))
            eta_values = np.linspace(-radius, radius, 81)
        objective = str(objective).lower()
        hamiltonian = self.hamiltonian()

        best_eta = None
        best_value = None
        for eta in np.asarray(eta_values, dtype=float):
            unitary = self.lang_firsov_unitary(float(eta))
            rotated = unitary.conj().T @ state
            singular_values = self.schmidt_spectrum(rotated)
            if objective in ("discarded", "discarded_weight", "weight"):
                value = discarded_weight(singular_values, rank)
            elif objective == "energy":
                value = self.rank_projected_energy(
                    rank, state=state, eta=float(eta), hamiltonian=hamiltonian
                )
            else:
                raise ValueError("objective must be 'discarded_weight' or 'energy'.")
            if best_value is None or value < best_value:
                best_value = float(value)
                best_eta = float(eta)
        return best_eta, best_value

    def report(
        self,
        *,
        ranks: tuple[int, ...] = (1, 2),
        nstates: int = 4,
        eta: float | None = None,
        eta_values: np.ndarray | None = None,
    ) -> HolsteinDimerReport:
        """Return exact, bare-rank, and Lang-Firsov-rank diagnostics."""
        energies, vectors = self.eigensystem(nstates=nstates)
        state = vectors[:, 0]
        hamiltonian = self.hamiltonian()

        if eta is None:
            eta, _ = self.optimize_lang_firsov_eta(
                state=state, rank=min(ranks), eta_values=eta_values
            )
        bare_s = self.schmidt_spectrum(state)
        unitary = self.lang_firsov_unitary(eta)
        lf_state = unitary.conj().T @ state
        lf_s = self.schmidt_spectrum(lf_state)

        ranks = tuple(int(rank) for rank in ranks)
        bare = RankDFrameReport(
            eta=None,
            singular_values=bare_s,
            discarded_weights={rank: discarded_weight(bare_s, rank) for rank in ranks},
            energies={
                rank: self.rank_projected_energy(rank, state=state, hamiltonian=hamiltonian)
                for rank in ranks
            },
        )
        lang_firsov = RankDFrameReport(
            eta=float(eta),
            singular_values=lf_s,
            discarded_weights={rank: discarded_weight(lf_s, rank) for rank in ranks},
            energies={
                rank: self.rank_projected_energy(
                    rank, state=state, eta=float(eta), hamiltonian=hamiltonian
                )
                for rank in ranks
            },
        )
        return HolsteinDimerReport(
            model=self,
            exact_energies=energies,
            bare=bare,
            lang_firsov=lang_firsov,
        )


def finite_difference_kinetic(grid: np.ndarray, *, mass: float = 1.0) -> np.ndarray:
    """Second-order finite-difference kinetic matrix with Dirichlet edges."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("grid must be a one-dimensional array with at least 3 points.")
    spacing = np.diff(grid)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("finite_difference_kinetic expects a uniform grid.")
    dx = float(spacing[0])
    prefactor = -0.5 / (float(mass) * dx * dx)
    laplacian = np.zeros((grid.size, grid.size), dtype=float)
    np.fill_diagonal(laplacian, -2.0)
    np.fill_diagonal(laplacian[1:], 1.0)
    np.fill_diagonal(laplacian[:, 1:], 1.0)
    return prefactor * laplacian


def sine_dvr_grid(npoints: int, qmax: float) -> np.ndarray:
    """Return interior sine-DVR points for dimensionless ``q`` on ``[-qmax, qmax]``."""
    npoints = int(npoints)
    qmax = float(qmax)
    if npoints < 1:
        raise ValueError("npoints must be at least 1.")
    if qmax <= 0:
        raise ValueError("qmax must be positive.")
    length = 2.0 * qmax
    return -qmax + length * np.arange(1, npoints + 1, dtype=float) / (npoints + 1)


def sine_dvr_kinetic(npoints: int, qmax: float) -> np.ndarray:
    """Return the sine-DVR matrix for ``-0.5 d^2/dq^2`` on ``[-qmax, qmax]``."""
    npoints = int(npoints)
    qmax = float(qmax)
    if npoints < 1:
        raise ValueError("npoints must be at least 1.")
    if qmax <= 0:
        raise ValueError("qmax must be positive.")
    length = 2.0 * qmax
    points = np.arange(1, npoints + 1, dtype=float)
    modes = np.arange(1, npoints + 1, dtype=float)
    transform = np.sqrt(2.0 / (npoints + 1)) * np.sin(
        np.pi * np.outer(points, modes) / (npoints + 1)
    )
    eigenvalues = 0.5 * (np.pi * modes / length) ** 2
    kinetic = (transform * eigenvalues) @ transform.T
    return 0.5 * (kinetic + kinetic.T)


def _lowest_eigensystem(hamiltonian: np.ndarray, nroots: int | None):
    if nroots is None:
        return eigh(hamiltonian)
    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be at least 1.")
    nroots = min(nroots, hamiltonian.shape[0])
    return eigh(hamiltonian, subset_by_index=(0, nroots - 1))


def _normalize_letta_holstein_order(order: str) -> str:
    name = str(order).lower().replace("_", "-")
    aliases = {
        "mode-first": "mode-first",
        "modes-first": "mode-first",
        "phonon-first": "mode-first",
        "phonons-first": "mode-first",
        "electronic-last": "mode-first",
        "electron-last": "mode-first",
        "electronic-first": "electronic-first",
        "electron-first": "electronic-first",
    }
    if name not in aliases:
        raise ValueError("order must be mode-first or electronic-first.")
    return aliases[name]


@dataclass(frozen=True)
class HolsteinDimerCoordinateNARG:
    """Coordinate-grid Holstein dimer with explicit conditional states.

    This is the NARG/Born-Huang form for the dimer.  The phonon coordinates
    ``(x1, x2)`` are the slow site, and the electronic dimer is diagonalized
    conditionally at each grid point:

        H_el(x1, x2) A[x1, x2] = A[x1, x2] eps[x1, x2].

    The final effective Hamiltonian is built in the moving basis
    ``A[p, i, a]`` with coefficients ``C[p, a]``.  Its wavefunction is
    ``psi[p, i] = sum_a A[p, i, a] C[p, a]``.
    """

    t: float = 0.2
    omega: float = 1.0
    g: float = 1.2
    ngrid: int = 21
    xmax: float = 6.0
    mass: float = 1.0

    def __post_init__(self):
        if self.ngrid < 3:
            raise ValueError("ngrid must be at least 3.")
        if self.xmax <= 0:
            raise ValueError("xmax must be positive.")
        if self.mass <= 0:
            raise ValueError("mass must be positive.")

    @property
    def electron_dim(self) -> int:
        return 2

    @property
    def phonon_dim(self) -> int:
        return int(self.ngrid) ** 2

    @property
    def dim(self) -> int:
        return self.electron_dim * self.phonon_dim

    def grid(self) -> np.ndarray:
        return np.linspace(-float(self.xmax), float(self.xmax), int(self.ngrid))

    def coordinate_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        grid = self.grid()
        x1, x2 = np.meshgrid(grid, grid, indexing="ij")
        return x1.reshape(-1), x2.reshape(-1)

    def phonon_potential(self) -> np.ndarray:
        x1, x2 = self.coordinate_mesh()
        return 0.5 * self.mass * self.omega**2 * (x1 * x1 + x2 * x2)

    def phonon_kinetic(self) -> np.ndarray:
        grid = self.grid()
        kinetic_1d = finite_difference_kinetic(grid, mass=self.mass)
        eye = np.eye(grid.size)
        return np.kron(kinetic_1d, eye) + np.kron(eye, kinetic_1d)

    def electronic_hamiltonian_at(self, x1: float, x2: float) -> np.ndarray:
        return np.array(
            [[self.g * x1, -self.t], [-self.t, self.g * x2]],
            dtype=float,
        )

    def conditional_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Return conditional electronic energies and vectors.

        Energies have shape ``(npoints, 2)``.  Vectors have shape
        ``(npoints, 2, 2)``, where the middle axis is the bare electronic site
        index and the last axis is the conditional-state index.
        """
        x1, x2 = self.coordinate_mesh()
        energies = np.empty((self.phonon_dim, self.electron_dim), dtype=float)
        vectors = np.empty((self.phonon_dim, self.electron_dim, self.electron_dim), dtype=float)
        for point, (q1, q2) in enumerate(zip(x1, x2)):
            local_energies, local_vectors = eigh(self.electronic_hamiltonian_at(q1, q2))
            energies[point] = local_energies
            vectors[point] = local_vectors
        self._align_conditional_state_signs(vectors)
        return energies, vectors

    def _align_conditional_state_signs(self, vectors: np.ndarray) -> None:
        n = int(self.ngrid)
        for i in range(n):
            for j in range(n):
                point = i * n + j
                if i == 0 and j == 0:
                    continue
                reference = (i - 1) * n + j if i > 0 else i * n + (j - 1)
                for state in range(self.electron_dim):
                    if np.dot(vectors[reference, :, state], vectors[point, :, state]) < 0:
                        vectors[point, :, state] *= -1.0

    def full_hamiltonian(self) -> np.ndarray:
        """Return the exact coordinate-grid Hamiltonian in ``(point, electron)`` order."""
        kinetic = self.phonon_kinetic()
        potential = self.phonon_potential()
        hamiltonian = np.kron(kinetic + np.diag(potential), np.eye(self.electron_dim))
        x1, x2 = self.coordinate_mesh()
        for point, (q1, q2) in enumerate(zip(x1, x2)):
            start = point * self.electron_dim
            stop = start + self.electron_dim
            hamiltonian[start:stop, start:stop] += self.electronic_hamiltonian_at(q1, q2)
        return hamiltonian

    def effective_hamiltonian(
        self,
        nstates_per_point: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the NARG Hamiltonian in the conditional-state basis."""
        nstates_per_point = int(nstates_per_point)
        if nstates_per_point < 1 or nstates_per_point > self.electron_dim:
            raise ValueError("nstates_per_point must be 1 or 2 for the Holstein dimer.")

        conditional_energies, conditional_vectors = self.conditional_states()
        a = conditional_vectors[:, :, :nstates_per_point]
        kinetic = self.phonon_kinetic()
        slow_potential = self.phonon_potential()

        # Project the slow kinetic energy between position-dependent electronic states:
        # H_kin[p,a,q,b] = T[p,q] <A[p,a] | A[q,b]>.
        hamiltonian = np.einsum("pq,pia,qib->paqb", kinetic, a, a, optimize=True)
        hamiltonian = hamiltonian.reshape(
            self.phonon_dim * nstates_per_point,
            self.phonon_dim * nstates_per_point,
        )
        diagonal = (slow_potential[:, None] + conditional_energies[:, :nstates_per_point]).reshape(-1)
        hamiltonian += np.diag(diagonal)
        return hamiltonian, conditional_energies, conditional_vectors

    def exact(self, nroots: int = 4) -> tuple[np.ndarray, np.ndarray]:
        return _lowest_eigensystem(self.full_hamiltonian(), nroots)

    def run(self, nstates_per_point: int = 1, nroots: int = 4) -> HolsteinDimerConditionalResult:
        hamiltonian, conditional_energies, conditional_vectors = self.effective_hamiltonian(
            nstates_per_point=nstates_per_point
        )
        energies, vectors = _lowest_eigensystem(hamiltonian, nroots)
        return HolsteinDimerConditionalResult(
            energies=energies,
            vectors=vectors,
            conditional_energies=conditional_energies,
            conditional_vectors=conditional_vectors[:, :, : int(nstates_per_point)],
            hamiltonian=hamiltonian,
            nstates_per_point=int(nstates_per_point),
        )

    def reconstruct_wavefunction(
        self,
        coefficients: np.ndarray,
        conditional_vectors: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct ``psi[point, electron]`` from ``A[point, i, a] C[point, a]``."""
        conditional_vectors = np.asarray(conditional_vectors)
        nstates_per_point = conditional_vectors.shape[2]
        coefficients = np.asarray(coefficients).reshape(self.phonon_dim, nstates_per_point)
        return np.einsum("pia,pa->pi", conditional_vectors, coefficients, optimize=True)


def holstein_chain_exact_hamiltonian(
    nsites: int,
    *,
    t: float = 0.2,
    omega: float = 1.0,
    g: float = 1.2,
    nphonon: int = 4,
) -> np.ndarray:
    """Dense exact one-electron Holstein-chain Hamiltonian for validation.

    This routine intentionally builds the exponentially large one-electron
    Hilbert space and is meant only for small tests/benchmarks.
    """
    nsites = int(nsites)
    nphonon = int(nphonon)
    if nsites < 1:
        raise ValueError("nsites must be at least 1.")
    if nphonon < 1:
        raise ValueError("nphonon must be at least 1.")

    phonon_dim = nphonon**nsites
    dim = nsites * phonon_dim
    hamiltonian = np.zeros((dim, dim), dtype=float)
    powers = np.asarray([nphonon**site for site in range(nsites)], dtype=int)

    def unpack(index: int) -> np.ndarray:
        occ = np.empty(nsites, dtype=int)
        value = int(index)
        for site in range(nsites):
            occ[site] = value % nphonon
            value //= nphonon
        return occ

    def state_index(electron_site: int, phonon_index: int) -> int:
        return int(electron_site) * phonon_dim + int(phonon_index)

    for phonon_index in range(phonon_dim):
        occupations = unpack(phonon_index)
        phonon_energy = omega * float(np.sum(occupations))
        for electron_site in range(nsites):
            col = state_index(electron_site, phonon_index)
            hamiltonian[col, col] += phonon_energy

            mode_occupation = occupations[electron_site]
            if mode_occupation + 1 < nphonon:
                row_phonon = phonon_index + powers[electron_site]
                row = state_index(electron_site, row_phonon)
                hamiltonian[row, col] += g * np.sqrt(mode_occupation + 1)
            if mode_occupation > 0:
                row_phonon = phonon_index - powers[electron_site]
                row = state_index(electron_site, row_phonon)
                hamiltonian[row, col] += g * np.sqrt(mode_occupation)

            if electron_site > 0:
                row = state_index(electron_site - 1, phonon_index)
                hamiltonian[row, col] += -t
            if electron_site + 1 < nsites:
                row = state_index(electron_site + 1, phonon_index)
                hamiltonian[row, col] += -t

    return hamiltonian


def holstein_chain_exact_energies(
    nsites: int,
    *,
    t: float = 0.2,
    omega: float = 1.0,
    g: float = 1.2,
    nphonon: int = 4,
    nroots: int = 4,
) -> np.ndarray:
    """Return exact dense one-electron Holstein-chain energies for small systems."""
    energies, _ = _lowest_eigensystem(
        holstein_chain_exact_hamiltonian(
            nsites, t=t, omega=omega, g=g, nphonon=nphonon
        ),
        nroots,
    )
    return energies


def _orthonormalize_columns(matrix: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis spanning the columns of ``matrix``."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    gram = _column_overlap_matrix(matrix)
    evals, evecs = eigh(gram)
    keep = evals > float(tol)
    if not np.any(keep):
        raise ValueError("conditional basis has zero numerical rank.")
    return matrix @ (evecs[:, keep] / np.sqrt(evals[keep]))


def _column_overlap_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian column-overlap matrix ``S = Q^dag Q``."""
    matrix = np.asarray(matrix, dtype=complex)
    gram = matrix.T.conj() @ matrix
    return 0.5 * (gram + gram.T.conj())


def _column_overlap_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalues of the column-overlap matrix."""
    return eigh(_column_overlap_matrix(matrix), eigvals_only=True)


def _combined_sector_hamiltonian(h0: np.ndarray, h1: np.ndarray) -> np.ndarray:
    h0 = np.asarray(h0)
    h1 = np.asarray(h1)
    out = np.zeros((h0.shape[0] + h1.shape[0], h0.shape[1] + h1.shape[1]), dtype=np.result_type(h0, h1, complex))
    out[: h0.shape[0], : h0.shape[1]] = h0
    out[h0.shape[0] :, h0.shape[1] :] = h1
    return out


def _majorana_x_operator(c: np.ndarray) -> np.ndarray:
    """Hermitian finite-Hilbert-space coupling channel ``c + c^dag``."""
    c = np.asarray(c)
    dim0, dim1 = c.shape
    out = np.zeros((dim0 + dim1, dim0 + dim1), dtype=np.result_type(c, complex))
    out[:dim0, dim0:] = c
    out[dim0:, :dim0] = c.T.conj()
    return out


def _majorana_p_operator(c: np.ndarray) -> np.ndarray:
    """Hermitian finite-Hilbert-space coupling channel ``i(c - c^dag)``."""
    c = np.asarray(c)
    dim0, dim1 = c.shape
    out = np.zeros((dim0 + dim1, dim0 + dim1), dtype=np.result_type(c, complex))
    out[:dim0, dim0:] = 1j * c
    out[dim0:, :dim0] = -1j * c.T.conj()
    return out


def _annihilation_operator(c: np.ndarray) -> np.ndarray:
    """Finite-Hilbert-space annihilation channel from one to zero sector."""
    c = np.asarray(c)
    dim0, dim1 = c.shape
    out = np.zeros((dim0 + dim1, dim0 + dim1), dtype=np.result_type(c, complex))
    out[:dim0, dim0:] = c
    return out


def _majorana_x_operator_from_square(c: np.ndarray) -> np.ndarray:
    """Hermitian quadrature ``c + c^dag`` for a square annihilation matrix."""
    c = np.asarray(c)
    return c + c.T.conj()


def _majorana_p_operator_from_square(c: np.ndarray) -> np.ndarray:
    """Hermitian quadrature ``i(c - c^dag)`` for a square annihilation matrix."""
    c = np.asarray(c)
    return 1j * (c - c.T.conj())


def _bit_count(value: int) -> int:
    return int(value).bit_count()


def _charge_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return int(left[0]) + int(right[0]), int(left[1]) + int(right[1])


def _charge_sub(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return int(left[0]) - int(right[0]), int(left[1]) - int(right[1])


def _charge_nonnegative(charge: tuple[int, int]) -> bool:
    return int(charge[0]) >= 0 and int(charge[1]) >= 0


def _charge_parity(charge: tuple[int, int]) -> int:
    return -1 if (int(charge[0]) + int(charge[1])) % 2 else 1


def _spin_delta(spin: str) -> tuple[int, int]:
    spin = str(spin).lower()
    if spin in ("up", "u", "alpha"):
        return (1, 0)
    if spin in ("down", "d", "beta"):
        return (0, 1)
    raise ValueError("spin must be 'up' or 'down'.")


def _combinations_bits(nsites: int, nelec: int) -> list[int]:
    nsites = int(nsites)
    nelec = int(nelec)
    if nelec < 0 or nelec > nsites:
        return []
    out = []
    for bits in range(1 << nsites):
        if _bit_count(bits) == nelec:
            out.append(bits)
    return out


def _resolve_spin_sector(
    nsites: int,
    nup: int | None,
    ndown: int | None,
) -> tuple[int, int]:
    nsites = int(nsites)
    if nup is None or ndown is None:
        if nsites % 2:
            raise ValueError("balanced half filling requires even nsites; pass nup/ndown explicitly.")
        nup = ndown = nsites // 2
    nup = int(nup)
    ndown = int(ndown)
    if nup < 0 or nup > nsites or ndown < 0 or ndown > nsites:
        raise ValueError("nup and ndown must be between 0 and nsites.")
    return nup, ndown


def _spinful_hh_basis(
    nsites: int,
    nphonon: int,
    nup: int | None,
    ndown: int | None,
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int], int, int]:
    nsites = int(nsites)
    nphonon = int(nphonon)
    nup, ndown = _resolve_spin_sector(nsites, nup, ndown)
    up_configs = _combinations_bits(nsites, nup)
    down_configs = _combinations_bits(nsites, ndown)
    phonon_dim = nphonon**nsites
    basis = []
    index = {}
    for up_bits in up_configs:
        for down_bits in down_configs:
            for phonon_index in range(phonon_dim):
                index[(up_bits, down_bits, phonon_index)] = len(basis)
                basis.append((up_bits, down_bits, phonon_index))
    return basis, index, nup, ndown


def _phonon_occupations(index: int, nsites: int, nphonon: int) -> np.ndarray:
    occupations = np.empty(int(nsites), dtype=int)
    value = int(index)
    for site in range(int(nsites)):
        occupations[site] = value % int(nphonon)
        value //= int(nphonon)
    return occupations


def _spin_orbital_bits(up_bits: int, down_bits: int, nsites: int) -> int:
    bits = 0
    for site in range(int(nsites)):
        if (int(up_bits) >> site) & 1:
            bits |= 1 << (2 * site)
        if (int(down_bits) >> site) & 1:
            bits |= 1 << (2 * site + 1)
    return bits


def _split_spin_orbital_bits(bits: int, nsites: int) -> tuple[int, int]:
    up_bits = 0
    down_bits = 0
    for site in range(int(nsites)):
        if (int(bits) >> (2 * site)) & 1:
            up_bits |= 1 << site
        if (int(bits) >> (2 * site + 1)) & 1:
            down_bits |= 1 << site
    return up_bits, down_bits


def _apply_cdag_c(bits: int, create_orbital: int, annihilate_orbital: int) -> tuple[int, int] | None:
    bits = int(bits)
    create_orbital = int(create_orbital)
    annihilate_orbital = int(annihilate_orbital)
    if ((bits >> annihilate_orbital) & 1) == 0:
        return None
    sign = -1 if _bit_count(bits & ((1 << annihilate_orbital) - 1)) % 2 else 1
    after_annihilate = bits & ~(1 << annihilate_orbital)
    if (after_annihilate >> create_orbital) & 1:
        return None
    sign *= -1 if _bit_count(after_annihilate & ((1 << create_orbital) - 1)) % 2 else 1
    return after_annihilate | (1 << create_orbital), sign


def spinful_holstein_hubbard_exact_hamiltonian(
    nsites: int,
    *,
    t: float = 0.2,
    omega: float = 1.0,
    g: float = 1.2,
    hubbard_u: float = 4.0,
    nphonon: int = 3,
    nup: int | None = None,
    ndown: int | None = None,
) -> np.ndarray:
    """Dense exact spinful Holstein-Hubbard Hamiltonian in a fixed spin sector."""
    nsites = int(nsites)
    nphonon = int(nphonon)
    if nsites < 1:
        raise ValueError("nsites must be at least 1.")
    if nphonon < 1:
        raise ValueError("nphonon must be at least 1.")
    basis, index, _nup, _ndown = _spinful_hh_basis(nsites, nphonon, nup, ndown)

    hamiltonian = np.zeros((len(basis), len(basis)), dtype=float)
    powers = np.asarray([nphonon**site for site in range(nsites)], dtype=int)

    for col, (up_bits, down_bits, phonon_index) in enumerate(basis):
        occupations = _phonon_occupations(phonon_index, nsites, nphonon)
        diagonal = omega * float(np.sum(occupations))
        for site in range(nsites):
            n_up_site = (up_bits >> site) & 1
            n_down_site = (down_bits >> site) & 1
            charge = n_up_site + n_down_site
            diagonal += hubbard_u * n_up_site * n_down_site

            if charge:
                mode_occupation = occupations[site]
                if mode_occupation + 1 < nphonon:
                    row_phonon = phonon_index + powers[site]
                    row = index[(up_bits, down_bits, row_phonon)]
                    hamiltonian[row, col] += g * charge * np.sqrt(mode_occupation + 1)
                if mode_occupation > 0:
                    row_phonon = phonon_index - powers[site]
                    row = index[(up_bits, down_bits, row_phonon)]
                    hamiltonian[row, col] += g * charge * np.sqrt(mode_occupation)

        hamiltonian[col, col] += diagonal

        bits = _spin_orbital_bits(up_bits, down_bits, nsites)
        for site in range(nsites - 1):
            for spin_offset in (0, 1):
                left = 2 * site + spin_offset
                right = 2 * (site + 1) + spin_offset
                for create_orbital, annihilate_orbital in ((left, right), (right, left)):
                    applied = _apply_cdag_c(bits, create_orbital, annihilate_orbital)
                    if applied is None:
                        continue
                    new_bits, sign = applied
                    new_up, new_down = _split_spin_orbital_bits(new_bits, nsites)
                    row = index.get((new_up, new_down, phonon_index))
                    if row is not None:
                        hamiltonian[row, col] += -t * sign

    return hamiltonian


def spinful_holstein_hubbard_exact_energies(
    nsites: int,
    *,
    t: float = 0.2,
    omega: float = 1.0,
    g: float = 1.2,
    hubbard_u: float = 4.0,
    nphonon: int = 3,
    nup: int | None = None,
    ndown: int | None = None,
    nroots: int = 4,
) -> np.ndarray:
    """Return exact dense spinful Holstein-Hubbard energies for small chains."""
    energies, _ = _lowest_eigensystem(
        spinful_holstein_hubbard_exact_hamiltonian(
            nsites,
            t=t,
            omega=omega,
            g=g,
            hubbard_u=hubbard_u,
            nphonon=nphonon,
            nup=nup,
            ndown=ndown,
        ),
        nroots,
    )
    return energies


@dataclass(frozen=True)
class SpinfulHHBipolaronDiagnostics:
    """Charge and local-pair diagnostics for a dense spinful HH eigenstate."""

    energy: float
    nup: int
    ndown: int
    density: np.ndarray
    double_occupancy: np.ndarray
    density_correlation: np.ndarray
    charge_correlation: np.ndarray
    momenta: np.ndarray
    charge_structure_factor: np.ndarray
    staggered_charge_structure: float
    pair_binding_energy: float | None = None


def _spinful_hh_density_observables(
    state: np.ndarray,
    basis: list[tuple[int, int, int]],
    nsites: int,
    nup: int,
    ndown: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    probabilities = np.abs(_normalized_state(state)) ** 2
    density = np.zeros(int(nsites), dtype=float)
    double_occupancy = np.zeros(int(nsites), dtype=float)
    density_correlation = np.zeros((int(nsites), int(nsites)), dtype=float)
    for probability, (up_bits, down_bits, _phonon_index) in zip(probabilities, basis):
        occupations = np.empty(int(nsites), dtype=float)
        for site in range(int(nsites)):
            n_up_site = (up_bits >> site) & 1
            n_down_site = (down_bits >> site) & 1
            occupations[site] = n_up_site + n_down_site
            double_occupancy[site] += probability * n_up_site * n_down_site
        density += probability * occupations
        density_correlation += probability * np.outer(occupations, occupations)

    average_density = float(int(nup) + int(ndown)) / float(nsites)
    charge_correlation = (
        density_correlation
        - average_density * density[:, None]
        - average_density * density[None, :]
        + average_density**2
    )
    sites = np.arange(int(nsites), dtype=float)
    momenta = 2.0 * np.pi * sites / float(nsites)
    phase_distances = sites[:, None] - sites[None, :]
    charge_structure_factor = np.empty(int(nsites), dtype=float)
    for q_index, momentum in enumerate(momenta):
        phases = np.exp(1j * momentum * phase_distances)
        charge_structure_factor[q_index] = float(
            (np.sum(phases * charge_correlation) / float(nsites)).real
        )
    staggered = (-1.0) ** phase_distances
    staggered_charge_structure = float(
        np.sum(staggered * charge_correlation).real / float(nsites)
    )
    return (
        density,
        double_occupancy,
        density_correlation,
        charge_correlation,
        momenta,
        charge_structure_factor,
        staggered_charge_structure,
    )


def spinful_hh_pair_binding_energy(
    nsites: int,
    *,
    t: float = 0.2,
    omega: float = 1.0,
    g: float = 1.2,
    hubbard_u: float = 4.0,
    nphonon: int = 3,
    nup: int | None = None,
    ndown: int | None = None,
) -> float:
    """Return pair binding for adding one up and one down electron.

    The diagnostic is

        Delta_pair = E(Nu+1, Nd+1) + E(Nu, Nd)
                     - E(Nu+1, Nd) - E(Nu, Nd+1).

    Negative values indicate that adding the two electrons together is
    energetically favored over adding them separately.
    """
    nsites = int(nsites)
    nup, ndown = _resolve_spin_sector(nsites, nup, ndown)
    if nup + 1 > nsites or ndown + 1 > nsites:
        raise ValueError("pair binding needs nup + 1 and ndown + 1 within nsites.")

    common = dict(
        nsites=nsites,
        t=t,
        omega=omega,
        g=g,
        hubbard_u=hubbard_u,
        nphonon=nphonon,
        nroots=1,
    )
    e00 = spinful_holstein_hubbard_exact_energies(**common, nup=nup, ndown=ndown)[0]
    e10 = spinful_holstein_hubbard_exact_energies(**common, nup=nup + 1, ndown=ndown)[0]
    e01 = spinful_holstein_hubbard_exact_energies(**common, nup=nup, ndown=ndown + 1)[0]
    e11 = spinful_holstein_hubbard_exact_energies(**common, nup=nup + 1, ndown=ndown + 1)[0]
    return float(e11 + e00 - e10 - e01)


def spinful_hh_bipolaron_diagnostics(
    nsites: int,
    *,
    t: float = 0.2,
    omega: float = 1.0,
    g: float = 1.2,
    hubbard_u: float = 4.0,
    nphonon: int = 3,
    nup: int | None = None,
    ndown: int | None = None,
    state_index: int = 0,
    include_pair_binding: bool = False,
) -> SpinfulHHBipolaronDiagnostics:
    """Dense small-system diagnostics for bipolaron/CDW tendencies."""
    nsites = int(nsites)
    nphonon = int(nphonon)
    nup, ndown = _resolve_spin_sector(nsites, nup, ndown)
    hamiltonian = spinful_holstein_hubbard_exact_hamiltonian(
        nsites,
        t=t,
        omega=omega,
        g=g,
        hubbard_u=hubbard_u,
        nphonon=nphonon,
        nup=nup,
        ndown=ndown,
    )
    state_index = int(state_index)
    if state_index < 0 or state_index >= hamiltonian.shape[0]:
        raise ValueError("state_index is outside the Hilbert-space dimension.")
    energies, vectors = eigh(hamiltonian, subset_by_index=(state_index, state_index))
    basis, _basis_index, _nup, _ndown = _spinful_hh_basis(nsites, nphonon, nup, ndown)
    (
        density,
        double_occupancy,
        density_correlation,
        charge_correlation,
        momenta,
        charge_structure_factor,
        staggered_charge_structure,
    ) = _spinful_hh_density_observables(
        vectors[:, 0],
        basis,
        nsites,
        nup,
        ndown,
    )
    pair_binding = None
    if include_pair_binding and nup + 1 <= nsites and ndown + 1 <= nsites:
        pair_binding = spinful_hh_pair_binding_energy(
            nsites,
            t=t,
            omega=omega,
            g=g,
            hubbard_u=hubbard_u,
            nphonon=nphonon,
            nup=nup,
            ndown=ndown,
        )
    return SpinfulHHBipolaronDiagnostics(
        energy=float(energies[0]),
        nup=nup,
        ndown=ndown,
        density=density,
        double_occupancy=double_occupancy,
        density_correlation=density_correlation,
        charge_correlation=charge_correlation,
        momenta=momenta,
        charge_structure_factor=charge_structure_factor,
        staggered_charge_structure=staggered_charge_structure,
        pair_binding_energy=pair_binding,
    )


@dataclass(frozen=True)
class SpinfulHHDressedSite:
    """Local polaron-dressed spinful Holstein-Hubbard site."""

    h: dict[tuple[int, int], np.ndarray]
    c: dict[str, dict[tuple[int, int], np.ndarray]]


@dataclass(frozen=True)
class SpinfulHHDressedPair:
    """Two adjacent dressed sites with separate left/right boundary operators."""

    h: dict[tuple[int, int], np.ndarray]
    c_left: dict[str, dict[tuple[int, int], np.ndarray]]
    c_right: dict[str, dict[tuple[int, int], np.ndarray]]


@dataclass(frozen=True)
class SpinfulHHBlock:
    """Spinful Holstein-Hubbard block sectors and boundary operators."""

    h: dict[tuple[int, int], np.ndarray]
    c_boundary: dict[str, dict[tuple[int, int], np.ndarray]]


@dataclass(frozen=True)
class SpinfulHHBlockLayout:
    """Flattened sector layout for matrix-free block operations."""

    charges: list[tuple[int, int]]
    offsets: dict[tuple[int, int], slice]
    dim: int


@dataclass(frozen=True)
class SpinfulHHNARGResult:
    """Result of the half-filled spinful Holstein-Hubbard block calculation."""

    energies: np.ndarray
    block: SpinfulHHBlock
    target: tuple[int, int]
    sector_dims: list[dict[tuple[int, int], int]]


@dataclass(frozen=True)
class SpinfulHHConditioningStep:
    """Diagnostics for one spinful coupling-conditioned NARG growth step."""

    site: int
    mode: str
    target_sector: tuple[int, int]
    local_site_dim: int
    site_branch_count: int
    states_per_branch: int
    conditional_dim: int
    raw_dim: int
    orthonormal_dim: int
    overlap_eigenvalues: np.ndarray


@dataclass(frozen=True)
class SpinfulHHCouplingNARGResult:
    """Result of spinful coupling-conditioned Holstein-Hubbard NARG."""

    energies: np.ndarray
    block: "SpinfulHHBlock"
    target: tuple[int, int]
    sector_dims: list[dict[tuple[int, int], int]]
    steps: list[SpinfulHHConditioningStep]


@dataclass(frozen=True)
class SpinfulHolsteinHubbardNARG:
    """Sector block calculation for spinful Holstein-Hubbard."""

    nsites: int
    t: float = 0.2
    omega: float = 1.0
    g: float = 1.2
    hubbard_u: float = 4.0
    nphonon: int = 4
    local_dim: int | None = None
    phonon_basis: str = "polaron"
    dvr_xmin: float = -6.0
    dvr_xmax: float = 6.0
    bond_dim: int = 64
    nup: int | None = None
    ndown: int | None = None

    def __post_init__(self):
        if self.nsites < 1:
            raise ValueError("nsites must be at least 1.")
        if self.nphonon < 1:
            raise ValueError("nphonon must be at least 1.")
        if self.local_dim is not None:
            if self.local_dim < 1:
                raise ValueError("local_dim must be at least 1.")
            if self.local_dim > self.nphonon:
                raise ValueError("local_dim cannot exceed nphonon.")
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be at least 1.")
        if self._phonon_basis_name() not in {"polaron", "fock", "dvr", "sine_dvr"}:
            raise ValueError("phonon_basis must be polaron, fock, dvr, or sine_dvr.")
        if self.dvr_xmax <= self.dvr_xmin:
            raise ValueError("dvr_xmax must be greater than dvr_xmin.")
        if self._phonon_basis_name() in {"dvr", "sine_dvr"} and self.local_dim is not None and self.local_dim != self.nphonon:
            raise ValueError("DVR phonon basis is primitive; use local_dim=None or local_dim=nphonon.")
        if (self.nup is None or self.ndown is None) and self.nsites % 2:
            raise ValueError("balanced half filling requires even nsites; pass nup/ndown explicitly.")

    def _phonon_basis_name(self) -> str:
        name = str(self.phonon_basis).lower().replace("-", "_")
        aliases = {
            "number": "fock",
            "fock": "fock",
            "polaron": "polaron",
            "local_polaron": "polaron",
            "x_dvr": "dvr",
            "xdvr": "dvr",
            "coordinate": "dvr",
            "coordinate_dvr": "dvr",
            "dvr": "dvr",
            "sine": "sine_dvr",
            "sine_dvr": "sine_dvr",
            "sinedvr": "sine_dvr",
            "box_dvr": "sine_dvr",
        }
        return aliases.get(name, name)

    @property
    def target(self) -> tuple[int, int]:
        if self.nup is None or self.ndown is None:
            return self.nsites // 2, self.nsites // 2
        return int(self.nup), int(self.ndown)

    def pair_binding_energy(
        self,
        *,
        nup: int | None = None,
        ndown: int | None = None,
    ) -> float:
        """Return NRG/NARG pair binding for adding one up/down electron."""
        if nup is None or ndown is None:
            nup, ndown = self.target
        nup = int(nup)
        ndown = int(ndown)
        if nup < 0 or ndown < 0 or nup + 1 > self.nsites or ndown + 1 > self.nsites:
            raise ValueError("pair binding needs nup/ndown >= 0 and nup + 1, ndown + 1 <= nsites.")

        e00 = replace(self, nup=nup, ndown=ndown).run(nroots=1).energies[0]
        e10 = replace(self, nup=nup + 1, ndown=ndown).run(nroots=1).energies[0]
        e01 = replace(self, nup=nup, ndown=ndown + 1).run(nroots=1).energies[0]
        e11 = replace(self, nup=nup + 1, ndown=ndown + 1).run(nroots=1).energies[0]
        return float(e11 + e00 - e10 - e01)

    def _local_charge_hamiltonian(self, charge: int, double: bool = False) -> np.ndarray:
        if self._phonon_basis_name() == "sine_dvr":
            x, kinetic = self._sine_dvr_grid_and_kinetic()
            potential = self.omega * (0.5 * x**2 - 0.5) + np.sqrt(2.0) * self.g * int(charge) * x
            if double:
                potential = potential + self.hubbard_u
            return 0.5 * self.omega * kinetic + np.diag(potential)

        b = boson_annihilation(self.nphonon)
        bdag = b.T.conj()
        num = bdag @ b
        x = b + bdag
        return self.omega * num + self.g * int(charge) * x + (self.hubbard_u if double else 0.0) * np.eye(self.nphonon)

    def _sine_dvr_grid_and_kinetic(self) -> tuple[np.ndarray, np.ndarray]:
        npts = int(self.nphonon)
        indices = np.arange(1, npts + 1, dtype=float)
        length = float(self.dvr_xmax - self.dvr_xmin)
        grid = float(self.dvr_xmin) + length * indices / (npts + 1)
        transform = np.sin(np.outer(indices, indices) * np.pi / (npts + 1))
        transform *= np.sqrt(2.0 / (npts + 1))
        kinetic_fbr = (np.pi * indices / length) ** 2
        kinetic = (transform.T * kinetic_fbr) @ transform
        return grid, 0.5 * (kinetic + kinetic.T)

    def _primitive_phonon_basis(self, keep: int) -> np.ndarray:
        basis = self._phonon_basis_name()
        if basis in {"fock", "sine_dvr"}:
            return np.eye(self.nphonon, keep)
        if basis == "dvr":
            x = boson_annihilation(self.nphonon)
            x = x + x.T.conj()
            grid, vectors = eigh(x)
            order = np.argsort(grid.real)
            return vectors[:, order[:keep]]
        raise ValueError("_primitive_phonon_basis is only for fock/dvr bases.")

    def dressed_site(self) -> SpinfulHHDressedSite:
        keep = self.nphonon if self.local_dim is None else int(self.local_dim)
        sectors = {
            (0, 0): self._local_charge_hamiltonian(0),
            (1, 0): self._local_charge_hamiltonian(1),
            (0, 1): self._local_charge_hamiltonian(1),
            (1, 1): self._local_charge_hamiltonian(2, double=True),
        }
        h = {}
        vectors = {}
        if self._phonon_basis_name() == "polaron":
            for charge, h_local in sectors.items():
                evals, evecs = eigh(h_local, subset_by_index=(0, keep - 1))
                h[charge] = np.diag(evals)
                vectors[charge] = evecs
        else:
            basis = self._primitive_phonon_basis(keep)
            for charge, h_local in sectors.items():
                h[charge] = basis.T.conj() @ h_local @ basis
                vectors[charge] = basis

        c = {"up": {}, "down": {}}
        transitions = {
            "up": [((1, 0), (0, 0), 1.0), ((1, 1), (0, 1), 1.0)],
            "down": [((0, 1), (0, 0), 1.0), ((1, 1), (1, 0), -1.0)],
        }
        for spin, entries in transitions.items():
            for source, target, sign in entries:
                c[spin][source] = sign * vectors[target].T.conj() @ vectors[source]
        return SpinfulHHDressedSite(h=h, c=c)

    def initial_block(self) -> SpinfulHHBlock:
        site = self.dressed_site()
        return SpinfulHHBlock(
            h={charge: value.copy() for charge, value in site.h.items()},
            c_boundary={
                spin: {charge: op.copy() for charge, op in ops.items()}
                for spin, ops in site.c.items()
            },
        )

    def _allowed_block_charge(self, charge: tuple[int, int]) -> bool:
        target = self.target
        return 0 <= charge[0] <= target[0] and 0 <= charge[1] <= target[1]

    def _truncate_sector(self, hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = min(int(self.bond_dim), hamiltonian.shape[0])
        evals, evecs = eigh(hamiltonian, subset_by_index=(0, keep - 1))
        return np.diag(evals), evecs

    def _branches_for_sector(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        branches = []
        for site_charge in site.h:
            block_charge = _charge_sub(sector, site_charge)
            if block_charge in block.h and _charge_nonnegative(block_charge):
                branches.append((block_charge, site_charge))
        return branches

    def _raw_sector_layout(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
    ) -> tuple[
        list[tuple[tuple[int, int], tuple[int, int]]],
        dict[tuple[tuple[int, int], tuple[int, int]], slice],
        int,
    ]:
        branches = self._branches_for_sector(block, site, sector)
        offsets = {}
        cursor = 0
        for branch in branches:
            block_charge, site_charge = branch
            dim = block.h[block_charge].shape[0] * site.h[site_charge].shape[0]
            offsets[branch] = slice(cursor, cursor + dim)
            cursor += dim
        return branches, offsets, cursor

    def _raw_sector_hamiltonian(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
    ) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[tuple[int, int], tuple[int, int]], slice]]:
        branches, offsets, cursor = self._raw_sector_layout(block, site, sector)
        hamiltonian = np.zeros((cursor, cursor), dtype=complex)
        for branch, span in offsets.items():
            block_charge, site_charge = branch
            hb = block.h[block_charge]
            hs = site.h[site_charge]
            hamiltonian[span, span] += np.kron(hb, np.eye(hs.shape[0]))
            hamiltonian[span, span] += np.kron(np.eye(hb.shape[0]), hs)

        for spin in ("up", "down"):
            delta = _spin_delta(spin)
            for ket_branch, ket_span in offsets.items():
                block_charge, site_charge = ket_branch
                source_site = site_charge
                target_site = _charge_sub(source_site, delta)
                target_block = _charge_add(block_charge, delta)
                bra_branch = (target_block, target_site)
                if bra_branch not in offsets:
                    continue
                if target_block not in block.c_boundary[spin]:
                    continue
                if source_site not in site.c[spin]:
                    continue
                parity = _charge_parity(block_charge)
                c_block_create = block.c_boundary[spin][target_block].T.conj()
                c_site_annihilate = site.c[spin][source_site]
                coupling = -self.t * parity * np.kron(c_block_create, c_site_annihilate)
                bra_span = offsets[bra_branch]
                hamiltonian[bra_span, ket_span] += coupling
                hamiltonian[ket_span, bra_span] += coupling.T.conj()

        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T.conj())
        return hamiltonian, branches, offsets

    def _apply_raw_sector_hamiltonian(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        branches: list[tuple[tuple[int, int], tuple[int, int]]],
        offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        raw_dim: int,
        vector: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros(raw_dim, dtype=complex)
        for branch in branches:
            block_charge, site_charge = branch
            span = offsets[branch]
            hb = block.h[block_charge]
            hs = site.h[site_charge]
            component = vector[span].reshape(hb.shape[0], hs.shape[0])
            result[span] += (hb @ component + component @ hs.T).reshape(-1)

        for spin in ("up", "down"):
            delta = _spin_delta(spin)
            for ket_branch, ket_span in offsets.items():
                block_charge, site_charge = ket_branch
                target_site = _charge_sub(site_charge, delta)
                target_block = _charge_add(block_charge, delta)
                bra_branch = (target_block, target_site)
                if bra_branch not in offsets:
                    continue
                if target_block not in block.c_boundary[spin]:
                    continue
                if site_charge not in site.c[spin]:
                    continue

                coeff = -self.t * _charge_parity(block_charge)
                c_block_create = block.c_boundary[spin][target_block].T.conj()
                c_site_annihilate = site.c[spin][site_charge]
                bra_span = offsets[bra_branch]

                ket_matrix = vector[ket_span].reshape(
                    block.h[block_charge].shape[0],
                    site.h[site_charge].shape[0],
                )
                result[bra_span] += (
                    coeff * (c_block_create @ ket_matrix @ c_site_annihilate.T)
                ).reshape(-1)

                bra_matrix = vector[bra_span].reshape(
                    block.h[target_block].shape[0],
                    site.h[target_site].shape[0],
                )
                result[ket_span] += (
                    coeff
                    * (c_block_create.T.conj() @ bra_matrix @ c_site_annihilate.conj())
                ).reshape(-1)
        return result

    def _project_raw_sector_hamiltonian(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        branches: list[tuple[tuple[int, int], tuple[int, int]]],
        offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        raw_dim: int,
        basis: np.ndarray,
    ) -> np.ndarray:
        applied = np.empty((raw_dim, basis.shape[1]), dtype=complex)
        for column in range(basis.shape[1]):
            applied[:, column] = self._apply_raw_sector_hamiltonian(
                block,
                site,
                branches,
                offsets,
                raw_dim,
                basis[:, column],
            )
        return basis.T.conj() @ applied

    def _apply_raw_boundary_annihilation(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        spin: str,
        source_offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        target_offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        target_dim: int,
        vector: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros(target_dim, dtype=complex)
        delta = _spin_delta(spin)
        for source_branch, source_span in source_offsets.items():
            block_charge, site_charge = source_branch
            target_site = _charge_sub(site_charge, delta)
            target_branch = (block_charge, target_site)
            if target_branch not in target_offsets or site_charge not in site.c[spin]:
                continue

            source_matrix = vector[source_span].reshape(
                block.h[block_charge].shape[0],
                site.h[site_charge].shape[0],
            )
            target_span = target_offsets[target_branch]
            c_site = site.c[spin][site_charge]
            result[target_span] += (
                _charge_parity(block_charge) * (source_matrix @ c_site.T)
            ).reshape(-1)
        return result

    def _project_raw_boundary_annihilation(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        spin: str,
        source_offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        target_offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        target_dim: int,
        source_basis: np.ndarray,
        target_basis: np.ndarray,
    ) -> np.ndarray:
        applied = np.empty((target_dim, source_basis.shape[1]), dtype=complex)
        for column in range(source_basis.shape[1]):
            applied[:, column] = self._apply_raw_boundary_annihilation(
                block,
                site,
                spin,
                source_offsets,
                target_offsets,
                target_dim,
                source_basis[:, column],
            )
        return target_basis.T.conj() @ applied

    def grow(self, block: SpinfulHHBlock) -> SpinfulHHBlock:
        site = self.dressed_site()
        all_sectors = sorted(
            {
                _charge_add(block_charge, site_charge)
                for block_charge in block.h
                for site_charge in site.h
                if self._allowed_block_charge(_charge_add(block_charge, site_charge))
            }
        )
        raw_data = {}
        new_h = {}
        rotations = {}
        for sector in all_sectors:
            h_raw, branches, offsets = self._raw_sector_hamiltonian(block, site, sector)
            if h_raw.size == 0:
                continue
            h_sector, rotation = self._truncate_sector(h_raw)
            new_h[sector] = h_sector
            rotations[sector] = rotation
            raw_data[sector] = (branches, offsets, h_raw.shape[0])

        new_c = {"up": {}, "down": {}}
        for spin in ("up", "down"):
            delta = _spin_delta(spin)
            for source_sector, source_h in new_h.items():
                target_sector = _charge_sub(source_sector, delta)
                if target_sector not in new_h:
                    continue
                source_branches, source_offsets, source_dim = raw_data[source_sector]
                target_branches, target_offsets, target_dim = raw_data[target_sector]
                c_raw = np.zeros((target_dim, source_dim), dtype=complex)
                for source_branch, source_span in source_offsets.items():
                    block_charge, site_charge = source_branch
                    target_site = _charge_sub(site_charge, delta)
                    target_branch = (block_charge, target_site)
                    if target_branch not in target_offsets or site_charge not in site.c[spin]:
                        continue
                    parity = _charge_parity(block_charge)
                    identity_block = np.eye(block.h[block_charge].shape[0])
                    c_site = site.c[spin][site_charge]
                    target_span = target_offsets[target_branch]
                    c_raw[target_span, source_span] += parity * np.kron(identity_block, c_site)
                new_c[spin][source_sector] = rotations[target_sector].T.conj() @ c_raw @ rotations[source_sector]

        return SpinfulHHBlock(h=new_h, c_boundary=new_c)

    def run(self, nroots: int = 4) -> SpinfulHHNARGResult:
        block = self.initial_block()
        sector_dims = [{charge: h.shape[0] for charge, h in block.h.items()}]
        for _site in range(1, int(self.nsites)):
            block = self.grow(block)
            sector_dims.append({charge: h.shape[0] for charge, h in block.h.items()})
        target = self.target
        if target not in block.h:
            raise ValueError(f"target sector {target} is absent from the final block.")
        energies = np.diag(block.h[target]).real[: int(nroots)]
        return SpinfulHHNARGResult(
            energies=energies,
            block=block,
            target=target,
            sector_dims=sector_dims,
        )


@dataclass(frozen=True)
class SpinfulHolsteinHubbardCouplingNARG(SpinfulHolsteinHubbardNARG):
    """Spinful HH NARG conditioned on one local hopping channel.

    With ``branch_rule="coupling"``, ``mode`` selects the Hermitian site
    operator whose eigenvectors define the local NARG branches.  The branch
    count is then the dressed local site dimension: four electronic occupations
    times the kept phonon states.  With ``branch_rule="electronic"``, the
    branches are only the four local electronic occupations.  With
    ``branch_rule="electronic_virtual"``, the four occupation branches are
    kept, but their block states are selected from a scalar-averaged
    second-order virtual boundary-hopping Hamiltonian.  With
    ``branch_rule="electronic_resolvent"``, the second-order correction is a
    matrix-valued branch-space resolvent.  With
    ``branch_rule="electronic_coupling"``, four electronic-only coupling
    eigenstates are used as branches, and the local phonon states are attached
    afterward.
    """

    mode: str = "x_charge"
    branch_rule: str = "coupling"
    states_per_branch: int | None = None
    orthonormal_tol: float = 1e-12
    conditional_solver: str = "auto"
    conditional_solver_threshold: int = 128
    conditional_solver_tol: float = 1e-10
    conditional_solver_maxiter: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.states_per_branch is not None and self.states_per_branch < 1:
            raise ValueError("states_per_branch must be at least 1.")
        if self.orthonormal_tol <= 0:
            raise ValueError("orthonormal_tol must be positive.")
        if self.conditional_solver not in {"auto", "dense", "iterative"}:
            raise ValueError("conditional_solver must be auto, dense, or iterative.")
        if self.conditional_solver_threshold < 1:
            raise ValueError("conditional_solver_threshold must be at least 1.")
        if self.conditional_solver_tol <= 0:
            raise ValueError("conditional_solver_tol must be positive.")
        if self.conditional_solver_maxiter is not None and self.conditional_solver_maxiter < 1:
            raise ValueError("conditional_solver_maxiter must be at least 1.")
        self._branch_rule_name()
        self._conditioning_channel(self.mode)

    def _branch_rule_name(self) -> str:
        name = str(self.branch_rule).lower().replace("-", "_")
        aliases = {
            "channel": "coupling",
            "coupling_channel": "coupling",
            "majorana": "coupling",
            "occupation": "electronic",
            "electron": "electronic",
            "electron_occupation": "electronic",
            "electronic_occupation": "electronic",
            "virtual": "electronic_virtual",
            "virtual_hopping": "electronic_virtual",
            "electronic_sw": "electronic_virtual",
            "occupation_virtual": "electronic_virtual",
            "resolvent": "electronic_resolvent",
            "matrix_resolvent": "electronic_resolvent",
            "feshbach": "electronic_resolvent",
            "electronic_feshbach": "electronic_resolvent",
            "x_electronic": "electronic_coupling",
            "electronic_x": "electronic_coupling",
            "electron_coupling": "electronic_coupling",
            "electronic_channel": "electronic_coupling",
            "xs": "electronic_coupling",
        }
        name = aliases.get(name, name)
        if name not in {
            "coupling",
            "electronic",
            "electronic_virtual",
            "electronic_resolvent",
            "electronic_coupling",
        }:
            raise ValueError(
                "branch_rule must be coupling, electronic, electronic_virtual, "
                "electronic_resolvent, or electronic_coupling."
            )
        return name

    def _states_per_branch(self, block_dim: int) -> int:
        keep = self.bond_dim if self.states_per_branch is None else self.states_per_branch
        return min(int(keep), int(block_dim))

    def _use_iterative_conditional_solver(self, block_dim: int, keep: int) -> bool:
        if self.conditional_solver == "dense":
            return False
        if keep >= block_dim:
            return False
        if self.conditional_solver == "iterative":
            return True
        return block_dim >= self.conditional_solver_threshold and keep <= block_dim // 2

    def _combined_site_data(
        self,
        site: SpinfulHHDressedSite,
    ) -> tuple[
        np.ndarray,
        dict[str, np.ndarray],
        dict[tuple[int, int], slice],
        list[tuple[int, int]],
    ]:
        charges = sorted(site.h)
        offsets = {}
        cursor = 0
        for charge in charges:
            dim = site.h[charge].shape[0]
            offsets[charge] = slice(cursor, cursor + dim)
            cursor += dim

        h_site = np.zeros((cursor, cursor), dtype=complex)
        for charge, span in offsets.items():
            h_site[span, span] = site.h[charge]

        c_site = {}
        for spin in ("up", "down"):
            op = np.zeros_like(h_site)
            for source_charge, c_block in site.c[spin].items():
                target_charge = _charge_sub(source_charge, _spin_delta(spin))
                if target_charge in offsets:
                    op[offsets[target_charge], offsets[source_charge]] = c_block
            c_site[spin] = op
        return h_site, c_site, offsets, charges

    def _block_layout(self, block: SpinfulHHBlock) -> SpinfulHHBlockLayout:
        charges = sorted(block.h)
        offsets = {}
        cursor = 0
        for charge in charges:
            dim = block.h[charge].shape[0]
            offsets[charge] = slice(cursor, cursor + dim)
            cursor += dim
        return SpinfulHHBlockLayout(charges=charges, offsets=offsets, dim=cursor)

    def _conditioning_channel(self, mode: str) -> str:
        mode = str(mode).lower()
        aliases = {
            "xu": "x_up",
            "xd": "x_down",
            "xup": "x_up",
            "xdown": "x_down",
            "pu": "p_up",
            "pd": "p_down",
            "charge": "x_charge",
            "spin": "x_spin",
        }
        valid = {
            "x_up",
            "x_down",
            "p_up",
            "p_down",
            "x_charge",
            "x_spin",
            "bilinear",
        }
        if mode in aliases:
            return aliases[mode]
        if mode in valid:
            return mode
        raise ValueError(
            "mode must select one local basis: x_up, x_down, p_up, p_down, "
            "x_charge, x_spin, or bilinear."
        )

    def _site_conditioning_vectors(
        self,
        site: SpinfulHHDressedSite,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        _h_site, c_site, _offsets, _charges = self._combined_site_data(site)
        x_up = _majorana_x_operator_from_square(c_site["up"])
        x_down = _majorana_x_operator_from_square(c_site["down"])
        p_up = _majorana_p_operator_from_square(c_site["up"])
        p_down = _majorana_p_operator_from_square(c_site["down"])
        operators = {
            "x_up": x_up,
            "x_down": x_down,
            "p_up": p_up,
            "p_down": p_down,
            "x_charge": x_up + x_down,
            "x_spin": x_up - x_down,
            "bilinear": 1j * x_up @ x_down,
        }
        operators["bilinear"] = 0.5 * (operators["bilinear"] + operators["bilinear"].T.conj())

        channel = self._conditioning_channel(self.mode)
        _evals, evecs = eigh(operators[channel])
        return [evecs[:, col] for col in range(evecs.shape[1])], c_site

    def _electronic_site_conditioning_vectors(
        self,
        site: SpinfulHHDressedSite,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray], list[tuple[int, int]]]:
        charges = sorted(site.h)
        charge_index = {charge: index for index, charge in enumerate(charges)}
        c_site = {spin: np.zeros((len(charges), len(charges)), dtype=complex) for spin in ("up", "down")}
        transitions = {
            "up": [((1, 0), (0, 0), 1.0), ((1, 1), (0, 1), 1.0)],
            "down": [((0, 1), (0, 0), 1.0), ((1, 1), (1, 0), -1.0)],
        }
        for spin, entries in transitions.items():
            for source, target, sign in entries:
                if source in charge_index and target in charge_index:
                    c_site[spin][charge_index[target], charge_index[source]] = sign

        x_up = _majorana_x_operator_from_square(c_site["up"])
        x_down = _majorana_x_operator_from_square(c_site["down"])
        p_up = _majorana_p_operator_from_square(c_site["up"])
        p_down = _majorana_p_operator_from_square(c_site["down"])
        operators = {
            "x_up": x_up,
            "x_down": x_down,
            "p_up": p_up,
            "p_down": p_down,
            "x_charge": x_up + x_down,
            "x_spin": x_up - x_down,
            "bilinear": 1j * x_up @ x_down,
        }
        operators["bilinear"] = 0.5 * (operators["bilinear"] + operators["bilinear"].T.conj())

        channel = self._conditioning_channel(self.mode)
        _evals, evecs = eigh(operators[channel])
        return [evecs[:, col] for col in range(evecs.shape[1])], c_site, charges

    def _conditional_block_matrix(
        self,
        block: SpinfulHHBlock,
        layout: SpinfulHHBlockLayout,
        c_site: dict[str, np.ndarray],
        site_vector: np.ndarray,
    ) -> np.ndarray:
        h_cond = np.zeros((layout.dim, layout.dim), dtype=complex)
        for charge, span in layout.offsets.items():
            h_cond[span, span] = block.h[charge]

        for spin in ("up", "down"):
            eta = np.vdot(site_vector, c_site[spin] @ site_vector)
            delta = _spin_delta(spin)
            for source_charge, c_matrix in block.c_boundary[spin].items():
                target_charge = _charge_sub(source_charge, delta)
                if source_charge not in layout.offsets or target_charge not in layout.offsets:
                    continue
                source_span = layout.offsets[source_charge]
                target_span = layout.offsets[target_charge]
                target_parity = _charge_parity(target_charge)
                h_cond[target_span, source_span] += (
                    -self.t * eta.conjugate() * target_parity * c_matrix
                )
                h_cond[source_span, target_span] += (
                    -self.t * eta * target_parity * c_matrix.T.conj()
                )
        return 0.5 * (h_cond + h_cond.T.conj())

    def _conditional_block_operator(
        self,
        block: SpinfulHHBlock,
        layout: SpinfulHHBlockLayout,
        c_site: dict[str, np.ndarray],
        site_vector: np.ndarray,
    ) -> LinearOperator:
        eta = {
            spin: np.vdot(site_vector, c_site[spin] @ site_vector)
            for spin in ("up", "down")
        }
        transitions = []
        for spin in ("up", "down"):
            delta = _spin_delta(spin)
            for source_charge, c_matrix in block.c_boundary[spin].items():
                target_charge = _charge_sub(source_charge, delta)
                if source_charge not in layout.offsets or target_charge not in layout.offsets:
                    continue
                transitions.append(
                    (
                        spin,
                        layout.offsets[source_charge],
                        layout.offsets[target_charge],
                        _charge_parity(target_charge),
                        c_matrix,
                        c_matrix.T.conj(),
                    )
                )

        def matvec(vector: np.ndarray) -> np.ndarray:
            result = np.zeros_like(vector, dtype=complex)
            for charge, span in layout.offsets.items():
                result[span] += block.h[charge] @ vector[span]
            for spin, source_span, target_span, target_parity, c_matrix, c_dag in transitions:
                result[target_span] += (
                    -self.t * eta[spin].conjugate() * target_parity * (c_matrix @ vector[source_span])
                )
                result[source_span] += (
                    -self.t * eta[spin] * target_parity * (c_dag @ vector[target_span])
                )
            return result

        return LinearOperator((layout.dim, layout.dim), matvec=matvec, dtype=np.complex128)

    def _lowest_conditional_block_vectors(
        self,
        block: SpinfulHHBlock,
        layout: SpinfulHHBlockLayout,
        c_site: dict[str, np.ndarray],
        site_vector: np.ndarray,
        keep: int,
    ) -> np.ndarray:
        block_dim = layout.dim
        keep = min(int(keep), int(block_dim))
        if keep < 1:
            raise ValueError("keep must be at least 1.")

        if self._use_iterative_conditional_solver(block_dim, keep):
            operator = self._conditional_block_operator(block, layout, c_site, site_vector)
            try:
                evals, vectors = eigsh(
                    operator,
                    k=keep,
                    which="SA",
                    tol=float(self.conditional_solver_tol),
                    maxiter=self.conditional_solver_maxiter,
                )
                order = np.argsort(evals.real)
                return vectors[:, order]
            except ArpackNoConvergence:
                if self.conditional_solver == "iterative":
                    raise

        h_cond = self._conditional_block_matrix(block, layout, c_site, site_vector)
        _evals, vectors = eigh(h_cond, subset_by_index=(0, keep - 1))
        return vectors

    def _lowest_sector_vectors(self, hamiltonian: np.ndarray, keep: int) -> np.ndarray:
        keep = min(int(keep), hamiltonian.shape[0])
        if keep < 1:
            raise ValueError("keep must be at least 1.")
        if keep == hamiltonian.shape[0]:
            _evals, vectors = eigh(hamiltonian)
            return vectors
        _evals, vectors = eigh(hamiltonian, subset_by_index=(0, keep - 1))
        return vectors

    def _virtual_hopping_resolvent_weights(
        self,
        site: SpinfulHHDressedSite,
        current_charge: tuple[int, int],
        virtual_charge: tuple[int, int],
        site_amplitude: np.ndarray,
        virtual_block_evals: np.ndarray,
        current_block_reference: float,
    ) -> np.ndarray:
        current_site_evals, current_site_vecs = eigh(site.h[current_charge])
        virtual_site_evals, virtual_site_vecs = eigh(site.h[virtual_charge])
        amplitude = virtual_site_vecs.T.conj() @ site_amplitude @ current_site_vecs
        strengths = np.abs(amplitude) ** 2
        weights = np.empty_like(virtual_block_evals, dtype=float)
        floor = max(float(self.conditional_solver_tol), 1e-12)
        for index, virtual_block_eval in enumerate(np.asarray(virtual_block_evals, dtype=float)):
            denominators = (
                float(virtual_block_eval)
                + virtual_site_evals[:, None].real
                - float(current_block_reference)
                - current_site_evals[None, :].real
            )
            signs = np.where(denominators < 0.0, -1.0, 1.0)
            denominators = np.where(np.abs(denominators) < floor, signs * floor, denominators)
            weights[index] = float(np.mean(np.sum(strengths / denominators, axis=0)).real)
        return weights

    def _virtual_hopping_corrected_block_hamiltonian(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
        site_charge: tuple[int, int],
    ) -> np.ndarray:
        block_charge = _charge_sub(sector, site_charge)
        h_current = np.array(block.h[block_charge], dtype=complex, copy=True)
        current_evals = eigh(h_current, eigvals_only=True)
        current_reference = float(np.min(current_evals.real))

        for spin in ("up", "down"):
            delta = _spin_delta(spin)

            virtual_site = _charge_sub(site_charge, delta)
            virtual_block = _charge_add(block_charge, delta)
            if (
                virtual_site in site.h
                and virtual_block in block.h
                and site_charge in site.c[spin]
                and virtual_block in block.c_boundary[spin]
            ):
                virtual_evals, virtual_vecs = eigh(block.h[virtual_block])
                weights = self._virtual_hopping_resolvent_weights(
                    site,
                    site_charge,
                    virtual_site,
                    site.c[spin][site_charge],
                    virtual_evals.real,
                    current_reference,
                )
                hop = block.c_boundary[spin][virtual_block].T.conj()
                dressed_hop = virtual_vecs.T.conj() @ hop
                h_current -= self.t**2 * (dressed_hop.T.conj() * weights) @ dressed_hop

            virtual_site = _charge_add(site_charge, delta)
            virtual_block = _charge_sub(block_charge, delta)
            if (
                virtual_site in site.h
                and virtual_block in block.h
                and virtual_site in site.c[spin]
                and block_charge in block.c_boundary[spin]
            ):
                virtual_evals, virtual_vecs = eigh(block.h[virtual_block])
                weights = self._virtual_hopping_resolvent_weights(
                    site,
                    site_charge,
                    virtual_site,
                    site.c[spin][virtual_site].T.conj(),
                    virtual_evals.real,
                    current_reference,
                )
                hop = block.c_boundary[spin][block_charge]
                dressed_hop = virtual_vecs.T.conj() @ hop
                h_current -= self.t**2 * (dressed_hop.T.conj() * weights) @ dressed_hop

        return 0.5 * (h_current + h_current.T.conj())

    def _lowest_virtual_hopping_block_vectors(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
        site_charge: tuple[int, int],
        keep: int,
    ) -> np.ndarray:
        h_cond = self._virtual_hopping_corrected_block_hamiltonian(
            block,
            site,
            sector,
            site_charge,
        )
        return self._lowest_sector_vectors(h_cond, keep)

    def _branch_resolvent_hamiltonian(
        self,
        h_raw: np.ndarray,
        offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        branch: tuple[tuple[int, int], tuple[int, int]],
    ) -> np.ndarray:
        span = offsets[branch]
        h_eff = np.array(h_raw[span, span], dtype=complex, copy=True)
        reference = float(np.min(eigh(h_eff, eigvals_only=True).real))
        floor = max(float(self.conditional_solver_tol), 1e-10)
        for virtual_branch, virtual_span in offsets.items():
            if virtual_branch == branch:
                continue
            coupling = h_raw[virtual_span, span]
            if np.linalg.norm(coupling) <= self.orthonormal_tol:
                continue
            h_virtual = h_raw[virtual_span, virtual_span]
            evals, evecs = eigh(h_virtual)
            denominators = evals.real - reference
            signs = np.where(denominators < 0.0, -1.0, 1.0)
            denominators = np.where(np.abs(denominators) < floor, signs * floor, denominators)
            dressed = evecs.T.conj() @ coupling
            h_eff -= (dressed.T.conj() * (1.0 / denominators)) @ dressed
        return 0.5 * (h_eff + h_eff.T.conj())

    def _electronic_resolvent_projector_for_sector(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
        offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        raw_dim: int,
    ) -> tuple[np.ndarray, int, int, int]:
        h_raw, _branches, _offsets = self._raw_sector_hamiltonian(block, site, sector)
        states_per_branch = self._states_per_branch(self._block_layout(block).dim)
        site_charges = sorted(site.h)
        local_site_dim = sum(site.h[charge].shape[0] for charge in site_charges)
        site_branch_count = len(site_charges)
        columns = []
        for site_charge in site_charges:
            block_charge = _charge_sub(sector, site_charge)
            branch = (block_charge, site_charge)
            if branch not in offsets:
                continue
            branch_dim = offsets[branch].stop - offsets[branch].start
            keep = min(states_per_branch * site.h[site_charge].shape[0], branch_dim)
            h_eff = self._branch_resolvent_hamiltonian(h_raw, offsets, branch)
            branch_vectors = self._lowest_sector_vectors(h_eff, keep)
            raw_span = offsets[branch]
            for column in range(branch_vectors.shape[1]):
                raw_vector = np.zeros(raw_dim, dtype=complex)
                raw_vector[raw_span] = branch_vectors[:, column]
                columns.append(raw_vector)
        if not columns:
            return (
                np.eye(raw_dim, dtype=complex),
                local_site_dim,
                site_branch_count,
                states_per_branch,
            )
        return np.column_stack(columns), local_site_dim, site_branch_count, states_per_branch

    def _electronic_projector_for_sector(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
        offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        raw_dim: int,
        *,
        virtual_hopping: bool = False,
    ) -> tuple[np.ndarray, int, int, int]:
        layout = self._block_layout(block)
        states_per_branch = self._states_per_branch(layout.dim)
        site_charges = sorted(site.h)
        local_site_dim = sum(site.h[charge].shape[0] for charge in site_charges)
        site_branch_count = len(site_charges)
        columns = []
        for site_charge in site_charges:
            block_charge = _charge_sub(sector, site_charge)
            branch = (block_charge, site_charge)
            if branch not in offsets:
                continue
            if virtual_hopping:
                block_vectors = self._lowest_virtual_hopping_block_vectors(
                    block,
                    site,
                    sector,
                    site_charge,
                    states_per_branch,
                )
            else:
                block_vectors = self._lowest_sector_vectors(
                    block.h[block_charge],
                    states_per_branch,
                )
            site_dim = site.h[site_charge].shape[0]
            raw_span = offsets[branch]
            for block_column in range(block_vectors.shape[1]):
                for site_column in range(site_dim):
                    site_vector = np.zeros(site_dim, dtype=complex)
                    site_vector[site_column] = 1.0
                    raw_vector = np.zeros(raw_dim, dtype=complex)
                    raw_vector[raw_span] = np.kron(
                        block_vectors[:, block_column],
                        site_vector,
                    )
                    columns.append(raw_vector)
        if not columns:
            return (
                np.eye(raw_dim, dtype=complex),
                local_site_dim,
                site_branch_count,
                states_per_branch,
            )
        return np.column_stack(columns), local_site_dim, site_branch_count, states_per_branch

    def _electronic_coupling_projector_for_sector(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
        branches: list[tuple[tuple[int, int], tuple[int, int]]],
        offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        raw_dim: int,
    ) -> tuple[np.ndarray, int, int, int]:
        layout = self._block_layout(block)
        electronic_vectors, electronic_c, site_charges = self._electronic_site_conditioning_vectors(site)
        _h_site, _site_c, site_offsets, _charges = self._combined_site_data(site)

        site_dims = {site.h[charge].shape[0] for charge in site_charges}
        if len(site_dims) != 1:
            raise ValueError("electronic_coupling branches require equal phonon dimensions in all local sectors.")
        site_dim = next(iter(site_dims))
        states_per_branch = self._states_per_branch(layout.dim)
        local_site_dim = sum(site.h[charge].shape[0] for charge in site_charges)
        site_branch_count = len(electronic_vectors)
        charge_index = {charge: index for index, charge in enumerate(site_charges)}
        columns = []
        for electronic_vector in electronic_vectors:
            block_vectors = self._lowest_conditional_block_vectors(
                block,
                layout,
                electronic_c,
                electronic_vector,
                states_per_branch,
            )
            for block_column in range(block_vectors.shape[1]):
                block_vector = block_vectors[:, block_column]
                for site_column in range(site_dim):
                    raw_vector = np.zeros(raw_dim, dtype=complex)
                    for branch in branches:
                        block_charge, site_charge = branch
                        coefficient = electronic_vector[charge_index[site_charge]]
                        if abs(coefficient) <= self.orthonormal_tol:
                            continue
                        raw_span = offsets[branch]
                        block_piece = block_vector[layout.offsets[block_charge]]
                        site_piece = np.zeros(site.h[site_charge].shape[0], dtype=complex)
                        site_piece[site_column] = coefficient
                        raw_vector[raw_span] = np.kron(block_piece, site_piece)
                    norm = np.linalg.norm(raw_vector)
                    if norm > self.orthonormal_tol:
                        columns.append(raw_vector / norm)
        if not columns:
            return np.eye(raw_dim, dtype=complex), local_site_dim, site_branch_count, states_per_branch
        return np.column_stack(columns), local_site_dim, site_branch_count, states_per_branch

    def _conditional_projector_for_sector(
        self,
        block: SpinfulHHBlock,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
        branches: list[tuple[tuple[int, int], tuple[int, int]]],
        offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        raw_dim: int,
    ) -> tuple[np.ndarray, int, int, int]:
        if self._branch_rule_name() == "electronic":
            return self._electronic_projector_for_sector(
                block,
                site,
                sector,
                offsets,
                raw_dim,
            )
        if self._branch_rule_name() == "electronic_virtual":
            return self._electronic_projector_for_sector(
                block,
                site,
                sector,
                offsets,
                raw_dim,
                virtual_hopping=True,
            )
        if self._branch_rule_name() == "electronic_resolvent":
            return self._electronic_resolvent_projector_for_sector(
                block,
                site,
                sector,
                offsets,
                raw_dim,
            )
        if self._branch_rule_name() == "electronic_coupling":
            return self._electronic_coupling_projector_for_sector(
                block,
                site,
                sector,
                branches,
                offsets,
                raw_dim,
            )

        layout = self._block_layout(block)
        site_vectors, c_site = self._site_conditioning_vectors(site)
        _h_site, _site_c, site_offsets, _site_charges = self._combined_site_data(site)

        states_per_branch = self._states_per_branch(layout.dim)
        local_site_dim = next(iter(site_vectors)).shape[0] if site_vectors else 0
        site_branch_count = len(site_vectors)
        columns = []
        for site_vector in site_vectors:
            block_vectors = self._lowest_conditional_block_vectors(
                block,
                layout,
                c_site,
                site_vector,
                states_per_branch,
            )
            for column in range(block_vectors.shape[1]):
                raw_vector = np.zeros(raw_dim, dtype=complex)
                block_vector = block_vectors[:, column]
                for branch in branches:
                    block_charge, site_charge = branch
                    raw_span = offsets[branch]
                    block_piece = block_vector[layout.offsets[block_charge]]
                    site_piece = site_vector[site_offsets[site_charge]]
                    raw_vector[raw_span] = np.kron(block_piece, site_piece)
                norm = np.linalg.norm(raw_vector)
                if norm > self.orthonormal_tol:
                    columns.append(raw_vector / norm)
        if not columns:
            return np.eye(raw_dim, dtype=complex), local_site_dim, site_branch_count, states_per_branch
        return np.column_stack(columns), local_site_dim, site_branch_count, states_per_branch

    def grow(self, block: SpinfulHHBlock) -> tuple[SpinfulHHBlock, list[SpinfulHHConditioningStep]]:
        site = self.dressed_site()
        all_sectors = sorted(
            {
                _charge_add(block_charge, site_charge)
                for block_charge in block.h
                for site_charge in site.h
                if self._allowed_block_charge(_charge_add(block_charge, site_charge))
            }
        )
        raw_data = {}
        new_h = {}
        rotations = {}
        steps = []
        for sector in all_sectors:
            branches, offsets, raw_dim = self._raw_sector_layout(block, site, sector)
            if raw_dim == 0:
                continue
            projector, local_site_dim, site_branch_count, states_per_branch = self._conditional_projector_for_sector(
                block, site, sector, branches, offsets, raw_dim
            )
            overlap_eigenvalues = _column_overlap_eigenvalues(projector)
            if self._branch_rule_name() in {"electronic", "electronic_virtual", "electronic_resolvent"}:
                q = projector
            else:
                q = _orthonormalize_columns(projector, tol=self.orthonormal_tol)
            h_eff = self._project_raw_sector_hamiltonian(
                block,
                site,
                branches,
                offsets,
                raw_dim,
                q,
            )
            h_eff = 0.5 * (h_eff + h_eff.T.conj())
            h_sector, v_eff = self._truncate_sector(h_eff)
            rotation = q @ v_eff
            new_h[sector] = h_sector
            rotations[sector] = rotation
            raw_data[sector] = (branches, offsets, raw_dim)
            steps.append(
                SpinfulHHConditioningStep(
                    site=-1,
                    mode=str(self.mode),
                    target_sector=sector,
                    local_site_dim=local_site_dim,
                    site_branch_count=site_branch_count,
                    states_per_branch=states_per_branch,
                    conditional_dim=projector.shape[1],
                    raw_dim=raw_dim,
                    orthonormal_dim=q.shape[1],
                    overlap_eigenvalues=overlap_eigenvalues,
                )
            )

        new_c = {"up": {}, "down": {}}
        for spin in ("up", "down"):
            delta = _spin_delta(spin)
            for source_sector, _source_h in new_h.items():
                target_sector = _charge_sub(source_sector, delta)
                if target_sector not in new_h:
                    continue
                _source_branches, source_offsets, _source_dim = raw_data[source_sector]
                _target_branches, target_offsets, target_dim = raw_data[target_sector]
                new_c[spin][source_sector] = self._project_raw_boundary_annihilation(
                    block,
                    site,
                    spin,
                    source_offsets,
                    target_offsets,
                    target_dim,
                    rotations[source_sector],
                    rotations[target_sector],
                )

        return SpinfulHHBlock(h=new_h, c_boundary=new_c), steps

    def run(self, nroots: int = 4) -> SpinfulHHCouplingNARGResult:
        block = self.initial_block()
        sector_dims = [{charge: h.shape[0] for charge, h in block.h.items()}]
        steps = []
        for site_index in range(1, int(self.nsites)):
            block, step_batch = self.grow(block)
            for step in step_batch:
                steps.append(
                    SpinfulHHConditioningStep(
                        site=site_index,
                        mode=step.mode,
                        target_sector=step.target_sector,
                        local_site_dim=step.local_site_dim,
                        site_branch_count=step.site_branch_count,
                        states_per_branch=step.states_per_branch,
                        conditional_dim=step.conditional_dim,
                        raw_dim=step.raw_dim,
                        orthonormal_dim=step.orthonormal_dim,
                        overlap_eigenvalues=step.overlap_eigenvalues,
                    )
                )
            sector_dims.append({charge: h.shape[0] for charge, h in block.h.items()})
        target = self.target
        if target not in block.h:
            raise ValueError(f"target sector {target} is absent from the final block.")
        energies = np.diag(block.h[target]).real[: int(nroots)]
        return SpinfulHHCouplingNARGResult(
            energies=energies,
            block=block,
            target=target,
            sector_dims=sector_dims,
            steps=steps,
        )


@dataclass(frozen=True)
class SpinfulHolsteinHubbardTwoSiteNARG(SpinfulHolsteinHubbardNARG):
    """Spinful HH block growth that adds two dressed sites before truncating."""

    pair_dim: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.pair_dim is not None and self.pair_dim < 1:
            raise ValueError("pair_dim must be at least 1.")

    def _pair_keep(self, dim: int) -> int:
        keep = self.bond_dim if self.pair_dim is None else int(self.pair_dim)
        return min(int(keep), int(dim))

    def _pair_branches_for_sector(
        self,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        branches = []
        for left_charge in site.h:
            right_charge = _charge_sub(sector, left_charge)
            if right_charge in site.h and _charge_nonnegative(right_charge):
                branches.append((left_charge, right_charge))
        return branches

    def _pair_raw_sector_layout(
        self,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
    ) -> tuple[
        list[tuple[tuple[int, int], tuple[int, int]]],
        dict[tuple[tuple[int, int], tuple[int, int]], slice],
        int,
    ]:
        branches = self._pair_branches_for_sector(site, sector)
        offsets = {}
        cursor = 0
        for branch in branches:
            left_charge, right_charge = branch
            dim = site.h[left_charge].shape[0] * site.h[right_charge].shape[0]
            offsets[branch] = slice(cursor, cursor + dim)
            cursor += dim
        return branches, offsets, cursor

    def _pair_raw_hamiltonian(
        self,
        site: SpinfulHHDressedSite,
        sector: tuple[int, int],
    ) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[tuple[int, int], tuple[int, int]], slice]]:
        branches, offsets, raw_dim = self._pair_raw_sector_layout(site, sector)
        hamiltonian = np.zeros((raw_dim, raw_dim), dtype=complex)
        for branch, span in offsets.items():
            left_charge, right_charge = branch
            hl = site.h[left_charge]
            hr = site.h[right_charge]
            hamiltonian[span, span] += np.kron(hl, np.eye(hr.shape[0]))
            hamiltonian[span, span] += np.kron(np.eye(hl.shape[0]), hr)

        for spin in ("up", "down"):
            delta = _spin_delta(spin)
            for ket_branch, ket_span in offsets.items():
                left_charge, right_charge = ket_branch
                target_right = _charge_sub(right_charge, delta)
                target_left = _charge_add(left_charge, delta)
                bra_branch = (target_left, target_right)
                if bra_branch not in offsets:
                    continue
                if target_left not in site.c[spin] or right_charge not in site.c[spin]:
                    continue
                c_left_create = site.c[spin][target_left].T.conj()
                c_right_annihilate = site.c[spin][right_charge]
                coupling = (
                    -self.t
                    * _charge_parity(left_charge)
                    * np.kron(c_left_create, c_right_annihilate)
                )
                bra_span = offsets[bra_branch]
                hamiltonian[bra_span, ket_span] += coupling
                hamiltonian[ket_span, bra_span] += coupling.T.conj()

        return 0.5 * (hamiltonian + hamiltonian.T.conj()), branches, offsets

    def _pair_raw_annihilation(
        self,
        site: SpinfulHHDressedSite,
        spin: str,
        side: str,
        source_offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        target_offsets: dict[tuple[tuple[int, int], tuple[int, int]], slice],
        target_dim: int,
    ) -> np.ndarray:
        source_dim = max((span.stop for span in source_offsets.values()), default=0)
        operator = np.zeros((target_dim, source_dim), dtype=complex)
        delta = _spin_delta(spin)
        for source_branch, source_span in source_offsets.items():
            left_charge, right_charge = source_branch
            if side == "left":
                target_left = _charge_sub(left_charge, delta)
                target_branch = (target_left, right_charge)
                if target_branch not in target_offsets or left_charge not in site.c[spin]:
                    continue
                block = np.kron(
                    site.c[spin][left_charge],
                    np.eye(site.h[right_charge].shape[0]),
                )
            elif side == "right":
                target_right = _charge_sub(right_charge, delta)
                target_branch = (left_charge, target_right)
                if target_branch not in target_offsets or right_charge not in site.c[spin]:
                    continue
                block = (
                    _charge_parity(left_charge)
                    * np.kron(
                        np.eye(site.h[left_charge].shape[0]),
                        site.c[spin][right_charge],
                    )
                )
            else:
                raise ValueError("side must be left or right.")
            operator[target_offsets[target_branch], source_span] += block
        return operator

    def dressed_pair(self) -> SpinfulHHDressedPair:
        site = self.dressed_site()
        all_sectors = sorted({_charge_add(left, right) for left in site.h for right in site.h})
        raw_data = {}
        h_pair = {}
        rotations = {}
        for sector in all_sectors:
            h_raw, branches, offsets = self._pair_raw_hamiltonian(site, sector)
            if h_raw.size == 0:
                continue
            keep = self._pair_keep(h_raw.shape[0])
            evals, evecs = eigh(h_raw, subset_by_index=(0, keep - 1))
            h_pair[sector] = np.diag(evals)
            rotations[sector] = evecs
            raw_data[sector] = (branches, offsets, h_raw.shape[0])

        c_pair = {"left": {"up": {}, "down": {}}, "right": {"up": {}, "down": {}}}
        for side in ("left", "right"):
            for spin in ("up", "down"):
                delta = _spin_delta(spin)
                for source_sector in h_pair:
                    target_sector = _charge_sub(source_sector, delta)
                    if target_sector not in h_pair:
                        continue
                    _source_branches, source_offsets, _source_dim = raw_data[source_sector]
                    _target_branches, target_offsets, target_dim = raw_data[target_sector]
                    c_raw = self._pair_raw_annihilation(
                        site,
                        spin,
                        side,
                        source_offsets,
                        target_offsets,
                        target_dim,
                    )
                    c_pair[side][spin][source_sector] = (
                        rotations[target_sector].T.conj() @ c_raw @ rotations[source_sector]
                    )

        return SpinfulHHDressedPair(
            h=h_pair,
            c_left=c_pair["left"],
            c_right=c_pair["right"],
        )

    def initial_pair_block(self) -> SpinfulHHBlock:
        pair = self.dressed_pair()
        allowed = {charge for charge in pair.h if self._allowed_block_charge(charge)}
        return SpinfulHHBlock(
            h={charge: pair.h[charge].copy() for charge in allowed},
            c_boundary={
                spin: {
                    charge: op.copy()
                    for charge, op in ops.items()
                    if charge in allowed and _charge_sub(charge, _spin_delta(spin)) in allowed
                }
                for spin, ops in pair.c_right.items()
            },
        )

    def _pair_block_raw_sector_layout(
        self,
        block: SpinfulHHBlock,
        pair: SpinfulHHDressedPair,
        sector: tuple[int, int],
    ) -> tuple[
        list[tuple[tuple[int, int], tuple[int, int]]],
        dict[tuple[tuple[int, int], tuple[int, int]], slice],
        int,
    ]:
        branches = []
        for pair_charge in pair.h:
            block_charge = _charge_sub(sector, pair_charge)
            if block_charge in block.h and _charge_nonnegative(block_charge):
                branches.append((block_charge, pair_charge))
        offsets = {}
        cursor = 0
        for branch in branches:
            block_charge, pair_charge = branch
            dim = block.h[block_charge].shape[0] * pair.h[pair_charge].shape[0]
            offsets[branch] = slice(cursor, cursor + dim)
            cursor += dim
        return branches, offsets, cursor

    def _pair_block_raw_hamiltonian(
        self,
        block: SpinfulHHBlock,
        pair: SpinfulHHDressedPair,
        sector: tuple[int, int],
    ) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[tuple[int, int], tuple[int, int]], slice]]:
        branches, offsets, raw_dim = self._pair_block_raw_sector_layout(block, pair, sector)
        hamiltonian = np.zeros((raw_dim, raw_dim), dtype=complex)
        for branch, span in offsets.items():
            block_charge, pair_charge = branch
            hb = block.h[block_charge]
            hp = pair.h[pair_charge]
            hamiltonian[span, span] += np.kron(hb, np.eye(hp.shape[0]))
            hamiltonian[span, span] += np.kron(np.eye(hb.shape[0]), hp)

        for spin in ("up", "down"):
            delta = _spin_delta(spin)
            for ket_branch, ket_span in offsets.items():
                block_charge, pair_charge = ket_branch
                target_pair = _charge_sub(pair_charge, delta)
                target_block = _charge_add(block_charge, delta)
                bra_branch = (target_block, target_pair)
                if bra_branch not in offsets:
                    continue
                if target_block not in block.c_boundary[spin] or pair_charge not in pair.c_left[spin]:
                    continue
                c_block_create = block.c_boundary[spin][target_block].T.conj()
                c_pair_annihilate = pair.c_left[spin][pair_charge]
                coupling = (
                    -self.t
                    * _charge_parity(block_charge)
                    * np.kron(c_block_create, c_pair_annihilate)
                )
                bra_span = offsets[bra_branch]
                hamiltonian[bra_span, ket_span] += coupling
                hamiltonian[ket_span, bra_span] += coupling.T.conj()

        return 0.5 * (hamiltonian + hamiltonian.T.conj()), branches, offsets

    def grow_pair(self, block: SpinfulHHBlock, pair: SpinfulHHDressedPair) -> SpinfulHHBlock:
        all_sectors = sorted(
            {
                _charge_add(block_charge, pair_charge)
                for block_charge in block.h
                for pair_charge in pair.h
                if self._allowed_block_charge(_charge_add(block_charge, pair_charge))
            }
        )
        raw_data = {}
        new_h = {}
        rotations = {}
        for sector in all_sectors:
            h_raw, branches, offsets = self._pair_block_raw_hamiltonian(block, pair, sector)
            if h_raw.size == 0:
                continue
            h_sector, rotation = self._truncate_sector(h_raw)
            new_h[sector] = h_sector
            rotations[sector] = rotation
            raw_data[sector] = (branches, offsets, h_raw.shape[0])

        new_c = {"up": {}, "down": {}}
        for spin in ("up", "down"):
            delta = _spin_delta(spin)
            for source_sector, _source_h in new_h.items():
                target_sector = _charge_sub(source_sector, delta)
                if target_sector not in new_h:
                    continue
                _source_branches, source_offsets, source_dim = raw_data[source_sector]
                _target_branches, target_offsets, target_dim = raw_data[target_sector]
                c_raw = np.zeros((target_dim, source_dim), dtype=complex)
                for source_branch, source_span in source_offsets.items():
                    block_charge, pair_charge = source_branch
                    target_pair = _charge_sub(pair_charge, delta)
                    target_branch = (block_charge, target_pair)
                    if target_branch not in target_offsets or pair_charge not in pair.c_right[spin]:
                        continue
                    identity_block = np.eye(block.h[block_charge].shape[0])
                    c_pair = pair.c_right[spin][pair_charge]
                    c_raw[target_offsets[target_branch], source_span] += (
                        _charge_parity(block_charge) * np.kron(identity_block, c_pair)
                    )
                new_c[spin][source_sector] = (
                    rotations[target_sector].T.conj() @ c_raw @ rotations[source_sector]
                )

        return SpinfulHHBlock(h=new_h, c_boundary=new_c)

    def run(self, nroots: int = 4) -> SpinfulHHNARGResult:
        if self.nsites == 1:
            return super().run(nroots=nroots)

        pair = self.dressed_pair()
        if self.nsites % 2 == 0:
            block = self.initial_pair_block()
            sites_done = 2
        else:
            block = self.initial_block()
            sites_done = 1

        sector_dims = [{charge: h.shape[0] for charge, h in block.h.items()}]
        while sites_done + 2 <= int(self.nsites):
            block = self.grow_pair(block, pair)
            sites_done += 2
            sector_dims.append({charge: h.shape[0] for charge, h in block.h.items()})
        while sites_done < int(self.nsites):
            block = self.grow(block)
            sites_done += 1
            sector_dims.append({charge: h.shape[0] for charge, h in block.h.items()})

        target = self.target
        if target not in block.h:
            raise ValueError(f"target sector {target} is absent from the final block.")
        energies = np.diag(block.h[target]).real[: int(nroots)]
        return SpinfulHHNARGResult(
            energies=energies,
            block=block,
            target=target,
            sector_dims=sector_dims,
        )


@dataclass(frozen=True)
class HolsteinChainNARG:
    """Recursive one-electron Holstein-chain NARG in a local Fock basis.

    This is the scalable counterpart to ``HolsteinDimerCoordinateNARG``.  It
    never builds the global phonon product grid during the NARG run.  Instead
    it carries two block sectors, zero electron and one electron, plus the
    renormalized boundary annihilation operator needed for the next hopping
    term.
    """

    nsites: int
    t: float = 0.2
    omega: float = 1.0
    g: float = 1.2
    nphonon: int = 8
    local_dim: int | None = None
    bond_dim: int = 32

    def __post_init__(self):
        if self.nsites < 1:
            raise ValueError("nsites must be at least 1.")
        if self.nphonon < 1:
            raise ValueError("nphonon must be at least 1.")
        if self.local_dim is not None:
            if self.local_dim < 1:
                raise ValueError("local_dim must be at least 1.")
            if self.local_dim > self.nphonon:
                raise ValueError("local_dim cannot exceed nphonon.")
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be at least 1.")

    def local_hamiltonians(self) -> tuple[np.ndarray, np.ndarray]:
        b = boson_annihilation(self.nphonon)
        bdag = b.T.conj()
        num = bdag @ b
        x = b + bdag
        h0 = self.omega * num
        h1 = self.omega * num + self.g * x
        return h0, h1

    def dressed_site(self) -> HolsteinDressedSite:
        """Diagonalize and truncate one Holstein site before block growth."""
        h0_local, h1_local = self.local_hamiltonians()
        keep = self.nphonon if self.local_dim is None else int(self.local_dim)
        e0, u0 = eigh(h0_local, subset_by_index=(0, keep - 1))
        e1, u1 = eigh(h1_local, subset_by_index=(0, keep - 1))
        c_bare = np.eye(self.nphonon)
        c = u0.T.conj() @ c_bare @ u1
        return HolsteinDressedSite(
            h0=np.diag(e0),
            h1=np.diag(e1),
            c=c,
        )

    def _truncate(self, hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = min(int(self.bond_dim), hamiltonian.shape[0])
        energies, vectors = eigh(hamiltonian, subset_by_index=(0, keep - 1))
        return np.diag(energies), vectors

    def initial_block(self) -> HolsteinChainBlock:
        site = self.dressed_site()
        return HolsteinChainBlock(h0=site.h0, h1=site.h1, c_boundary=site.c)

    def grow(self, block: HolsteinChainBlock) -> HolsteinChainBlock:
        site = self.dressed_site()
        eye0_block = np.eye(block.h0.shape[0])
        eye1_block = np.eye(block.h1.shape[0])
        eye0_site = np.eye(site.h0.shape[0])
        eye1_site = np.eye(site.h1.shape[0])

        h0_raw = np.kron(block.h0, eye0_site) + np.kron(eye0_block, site.h0)

        h1_left = np.kron(block.h1, eye0_site) + np.kron(eye1_block, site.h0)
        h1_right = np.kron(block.h0, eye1_site) + np.kron(eye0_block, site.h1)
        hop = -self.t * np.kron(block.c_boundary.T.conj(), site.c)
        h1_raw = np.block([[h1_left, hop], [hop.T.conj(), h1_right]])

        h0, u0 = self._truncate(h0_raw)
        h1, u1 = self._truncate(h1_raw)

        left_cols = h1_left.shape[1]
        c_raw = np.zeros((h0_raw.shape[0], h1_raw.shape[0]), dtype=complex)
        c_raw[:, left_cols:] = np.kron(eye0_block, site.c)
        c_boundary = u0.T.conj() @ c_raw @ u1

        return HolsteinChainBlock(h0=h0, h1=h1, c_boundary=c_boundary)

    def run(self, nroots: int = 4) -> HolsteinChainNARGResult:
        block = self.initial_block()
        sector_dims = [(block.h0.shape[0], block.h1.shape[0])]
        for _site in range(1, int(self.nsites)):
            block = self.grow(block)
            sector_dims.append((block.h0.shape[0], block.h1.shape[0]))
        energies = np.diag(block.h1).real[: int(nroots)]
        return HolsteinChainNARGResult(
            energies=energies,
            block=block,
            sector_dims=sector_dims,
        )


@dataclass(frozen=True)
class HolsteinChainAdiabaticNARG(HolsteinChainNARG):
    """Holstein chain grown in the explicit NARG conditional-basis form.

    At each step the incoming dressed site has two local sectors: no electron
    and one electron.  The old block is diagonalized conditionally for each
    incoming local state, giving a projector ``Q`` from the conditional NARG
    basis into the raw enlarged product basis.  The enlarged Hamiltonian is
    then solved in ``Q^dag H Q``.

    For the pure one-electron Holstein chain the only inter-site term is
    hopping, which is off-diagonal in the incoming-site charge.  Therefore the
    conditional block Hamiltonians are just the old block Hamiltonians plus a
    scalar local energy.  This class still follows the NARG data flow; it also
    makes clear why this minimal model needs either local polaron dressing,
    diagonal inter-site couplings, or more particles for a non-trivial
    conditional block update.
    """

    def _conditional_projector_one_electron(
        self,
        block: HolsteinChainBlock,
        site: HolsteinDressedSite,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        dim0 = block.h0.shape[0]
        dim1 = block.h1.shape[0]
        d0 = site.h0.shape[0]
        d1 = site.h1.shape[0]
        raw_dim = dim1 * d0 + dim0 * d1

        top_keep = min(int(self.bond_dim), dim1)
        bottom_keep = min(int(self.bond_dim), dim0)
        conditional_dim = d0 * top_keep + d1 * bottom_keep
        projector = np.zeros((raw_dim, conditional_dim), dtype=complex)

        col = 0
        for local_state in range(d0):
            # Conditional diagonalization of H_B^(1) + eps_0(local_state).
            # The block is already in its retained eigenbasis, so the
            # conditional eigenvectors are the selected unit columns.
            for state in range(top_keep):
                raw_index = state * d0 + local_state
                projector[raw_index, col] = 1.0
                col += 1

        top_raw_dim = dim1 * d0
        for local_state in range(d1):
            # Conditional diagonalization of H_B^(0) + eps_1(local_state).
            for state in range(bottom_keep):
                raw_index = top_raw_dim + state * d1 + local_state
                projector[raw_index, col] = 1.0
                col += 1

        return projector, (top_keep, bottom_keep)

    def grow(self, block: HolsteinChainBlock) -> tuple[HolsteinChainBlock, HolsteinAdiabaticStep]:
        site = self.dressed_site()
        eye0_block = np.eye(block.h0.shape[0])
        eye1_block = np.eye(block.h1.shape[0])
        eye0_site = np.eye(site.h0.shape[0])
        eye1_site = np.eye(site.h1.shape[0])

        h0_raw = np.kron(block.h0, eye0_site) + np.kron(eye0_block, site.h0)

        h1_left = np.kron(block.h1, eye0_site) + np.kron(eye1_block, site.h0)
        h1_right = np.kron(block.h0, eye1_site) + np.kron(eye0_block, site.h1)
        hop = -self.t * np.kron(block.c_boundary.T.conj(), site.c)
        h1_raw = np.block([[h1_left, hop], [hop.T.conj(), h1_right]])

        h0, u0 = self._truncate(h0_raw)

        projector, states_per_branch = self._conditional_projector_one_electron(block, site)
        h1_eff = projector.T.conj() @ h1_raw @ projector
        h1, v_eff = self._truncate(h1_eff)
        u1_raw = projector @ v_eff

        left_cols = h1_left.shape[1]
        c_raw = np.zeros((h0_raw.shape[0], h1_raw.shape[0]), dtype=complex)
        c_raw[:, left_cols:] = np.kron(eye0_block, site.c)
        c_boundary = u0.T.conj() @ c_raw @ u1_raw

        new_block = HolsteinChainBlock(h0=h0, h1=h1, c_boundary=c_boundary)
        step = HolsteinAdiabaticStep(
            site=-1,
            conditional_dim=h1_eff.shape[0],
            raw_dim=h1_raw.shape[0],
            states_per_branch=states_per_branch,
        )
        return new_block, step

    def run(self, nroots: int = 4) -> HolsteinAdiabaticNARGResult:
        block = self.initial_block()
        sector_dims = [(block.h0.shape[0], block.h1.shape[0])]
        steps = []
        for site_index in range(1, int(self.nsites)):
            block, step = self.grow(block)
            steps.append(
                HolsteinAdiabaticStep(
                    site=site_index,
                    conditional_dim=step.conditional_dim,
                    raw_dim=step.raw_dim,
                    states_per_branch=step.states_per_branch,
                )
            )
            sector_dims.append((block.h0.shape[0], block.h1.shape[0]))
        energies = np.diag(block.h1).real[: int(nroots)]
        return HolsteinAdiabaticNARGResult(
            energies=energies,
            block=block,
            sector_dims=sector_dims,
            steps=steps,
        )


@dataclass(frozen=True)
class HolsteinChainCouplingNARG(HolsteinChainNARG):
    """Holstein chain NARG conditioned on the site hopping operator.

    The dressed local annihilation matrix ``c_s`` maps one-electron local
    states to zero-electron local states, so it is rectangular/nilpotent rather
    than an ordinary Hermitian observable.  This class uses the finite
    Hilbert-space Hermitian channels

        X_s = c_s + c_s^dagger
        P_s = i(c_s - c_s^dagger)

    and currently chooses the eigenbasis of ``X_s``.  The hopping interaction
    is then evaluated in the quadrature form

        -t(c_B^dag c_s + c_s^dag c_B)
        = -(t/2)(X_B X_s + P_B P_s).

    For each site vector it diagonalizes the block Hamiltonian

        H_B - (t/2)(<X_s> X_B + <P_s> P_B),

    then projects ``|phi_B(lambda)> |chi_s(lambda)>`` back to the physical
    one-electron sector before solving the exact enlarged Hamiltonian in that
    adapted subspace.
    """

    states_per_branch: int | None = None
    orthonormal_tol: float = 1e-12

    def __post_init__(self):
        super().__post_init__()
        if self.states_per_branch is not None and self.states_per_branch < 1:
            raise ValueError("states_per_branch must be at least 1.")
        if self.orthonormal_tol <= 0:
            raise ValueError("orthonormal_tol must be positive.")

    def _raw_hamiltonians(
        self,
        block: HolsteinChainBlock,
        site: HolsteinDressedSite,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        eye0_block = np.eye(block.h0.shape[0])
        eye1_block = np.eye(block.h1.shape[0])
        eye0_site = np.eye(site.h0.shape[0])
        eye1_site = np.eye(site.h1.shape[0])

        h0_raw = np.kron(block.h0, eye0_site) + np.kron(eye0_block, site.h0)
        h1_left = np.kron(block.h1, eye0_site) + np.kron(eye1_block, site.h0)
        h1_right = np.kron(block.h0, eye1_site) + np.kron(eye0_block, site.h1)
        hop = -self.t * np.kron(block.c_boundary.T.conj(), site.c)
        h1_raw = np.block([[h1_left, hop], [hop.T.conj(), h1_right]])

        c_raw = np.zeros((h0_raw.shape[0], h1_raw.shape[0]), dtype=complex)
        c_raw[:, h1_left.shape[1] :] = np.kron(eye0_block, site.c)
        return h0_raw, h1_raw, c_raw, h1_left.shape[1]

    def _coupling_conditioned_projector(
        self,
        block: HolsteinChainBlock,
        site: HolsteinDressedSite,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        dim0 = block.h0.shape[0]
        dim1 = block.h1.shape[0]
        d0 = site.h0.shape[0]
        d1 = site.h1.shape[0]
        raw_dim = dim1 * d0 + dim0 * d1

        h_block = _combined_sector_hamiltonian(block.h0, block.h1)
        x_block = _majorana_x_operator(block.c_boundary)
        p_block = _majorana_p_operator(block.c_boundary)
        x_site = _majorana_x_operator(site.c)
        p_site = _majorana_p_operator(site.c)
        site_eigenvalues, site_vectors = eigh(x_site)
        site_annihilation_expectations = np.empty(site_eigenvalues.shape, dtype=complex)
        site_p_expectations = np.empty(site_eigenvalues.shape, dtype=complex)

        keep_per_branch = min(
            int(self.bond_dim if self.states_per_branch is None else self.states_per_branch),
            h_block.shape[0],
        )
        columns = []
        for site_index, site_vector in enumerate(site_vectors.T):
            site_zero = site_vector[:d0]
            site_one = site_vector[d0:]
            eta = np.vdot(site_zero, site.c @ site_one)
            site_annihilation_expectations[site_index] = eta
            x_expectation = np.vdot(site_vector, x_site @ site_vector)
            p_expectation = np.vdot(site_vector, p_site @ site_vector)
            site_p_expectations[site_index] = p_expectation
            h_cond = h_block - 0.5 * self.t * (
                x_expectation * x_block + p_expectation * p_block
            )
            h_cond = 0.5 * (h_cond + h_cond.T.conj())
            _, block_vectors = eigh(h_cond, subset_by_index=(0, keep_per_branch - 1))
            for branch in range(block_vectors.shape[1]):
                block_vector = block_vectors[:, branch]
                block_zero = block_vector[:dim0]
                block_one = block_vector[dim0:]
                raw_vector = np.empty(raw_dim, dtype=complex)
                raw_vector[: dim1 * d0] = np.kron(block_one, site_zero)
                raw_vector[dim1 * d0 :] = np.kron(block_zero, site_one)
                norm = np.linalg.norm(raw_vector)
                if norm > self.orthonormal_tol:
                    columns.append(raw_vector / norm)

        if not columns:
            raise ValueError("coupling-conditioned basis is empty.")
        projector = np.column_stack(columns)
        return (
            projector,
            site_eigenvalues,
            site_annihilation_expectations,
            site_p_expectations,
            keep_per_branch,
        )

    def grow(self, block: HolsteinChainBlock) -> tuple[HolsteinChainBlock, HolsteinAdiabaticStep]:
        site = self.dressed_site()
        h0_raw, h1_raw, c_raw, _left_cols = self._raw_hamiltonians(block, site)

        h0, u0 = self._truncate(h0_raw)
        (
            projector,
            site_eigenvalues,
            site_annihilation_expectations,
            site_p_expectations,
            keep_per_branch,
        ) = self._coupling_conditioned_projector(block, site)
        overlap_eigenvalues = _column_overlap_eigenvalues(projector)
        q = _orthonormalize_columns(projector, tol=self.orthonormal_tol)
        h1_eff = q.T.conj() @ h1_raw @ q
        h1, v_eff = self._truncate(h1_eff)
        u1_raw = q @ v_eff

        c_boundary = u0.T.conj() @ c_raw @ u1_raw
        new_block = HolsteinChainBlock(h0=h0, h1=h1, c_boundary=c_boundary)
        step = HolsteinAdiabaticStep(
            site=-1,
            conditional_dim=projector.shape[1],
            raw_dim=h1_raw.shape[0],
            states_per_branch=(keep_per_branch, keep_per_branch),
            orthonormal_dim=q.shape[1],
            site_eigenvalues=site_eigenvalues,
            site_annihilation_expectations=site_annihilation_expectations,
            site_p_expectations=site_p_expectations,
            overlap_eigenvalues=overlap_eigenvalues,
        )
        return new_block, step

    def run(self, nroots: int = 4) -> HolsteinAdiabaticNARGResult:
        block = self.initial_block()
        sector_dims = [(block.h0.shape[0], block.h1.shape[0])]
        steps = []
        for site_index in range(1, int(self.nsites)):
            block, step = self.grow(block)
            steps.append(
                HolsteinAdiabaticStep(
                    site=site_index,
                    conditional_dim=step.conditional_dim,
                    raw_dim=step.raw_dim,
                    states_per_branch=step.states_per_branch,
                    orthonormal_dim=step.orthonormal_dim,
                    site_eigenvalues=step.site_eigenvalues,
                    site_annihilation_expectations=step.site_annihilation_expectations,
                    site_p_expectations=step.site_p_expectations,
                    overlap_eigenvalues=step.overlap_eigenvalues,
                )
            )
            sector_dims.append((block.h0.shape[0], block.h1.shape[0]))
        energies = np.diag(block.h1).real[: int(nroots)]
        return HolsteinAdiabaticNARGResult(
            energies=energies,
            block=block,
            sector_dims=sector_dims,
            steps=steps,
        )


@dataclass(frozen=True)
class HolsteinElectronicFirstNARG:
    """One-electron Holstein NARG ordered as electronic structure plus modes.

    The first NARG site is the full one-electron electronic Hamiltonian on the
    chain.  Subsequent NARG sites are phonon modes.  During mode growth the
    algorithm carries the projected electronic density operators ``n_i`` so
    adding mode ``i`` only needs

    ``H_new = H_block + omega b_i^dag b_i + g n_i (b_i + b_i^dag)``.

    This is the electronic-first active-mode ordering discussed for comparing
    conditional NARG against real-space MPS/DMRG.
    """

    nsites: int
    t: float = 0.2
    omega: float = 1.0
    g: float = 1.2
    nphonon: int = 8
    local_dim: int | None = None
    bond_dim: int = 32
    mode_order: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.nsites < 1:
            raise ValueError("nsites must be at least 1.")
        if self.nphonon < 1:
            raise ValueError("nphonon must be at least 1.")
        if self.local_dim is not None:
            if self.local_dim < 1:
                raise ValueError("local_dim must be at least 1.")
            if self.local_dim > self.nphonon:
                raise ValueError("local_dim cannot exceed nphonon.")
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be at least 1.")
        if self.mode_order is not None:
            order = tuple(int(mode) for mode in self.mode_order)
            if sorted(order) != list(range(int(self.nsites))):
                raise ValueError("mode_order must be a permutation of range(nsites).")

    @property
    def active_mode_dim(self) -> int:
        return self.nphonon if self.local_dim is None else int(self.local_dim)

    def electronic_hamiltonian(self) -> np.ndarray:
        hamiltonian = np.zeros((int(self.nsites), int(self.nsites)), dtype=float)
        for site in range(int(self.nsites) - 1):
            hamiltonian[site, site + 1] = -float(self.t)
            hamiltonian[site + 1, site] = -float(self.t)
        return hamiltonian

    def electronic_density_operators(self) -> list[np.ndarray]:
        operators = []
        for site in range(int(self.nsites)):
            op = np.zeros((int(self.nsites), int(self.nsites)), dtype=float)
            op[site, site] = 1.0
            operators.append(op)
        return operators

    def phonon_operators(self) -> tuple[np.ndarray, np.ndarray]:
        dim = self.active_mode_dim
        b = boson_annihilation(dim)
        bdag = b.T.conj()
        return bdag @ b, b + bdag

    def _truncate(self, hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = min(int(self.bond_dim), hamiltonian.shape[0])
        energies, vectors = eigh(hamiltonian, subset_by_index=(0, keep - 1))
        return energies, vectors

    def run(self, nroots: int = 4) -> HolsteinElectronicFirstResult:
        block_h = self.electronic_hamiltonian().astype(complex)
        density_ops = [op.astype(complex) for op in self.electronic_density_operators()]
        mode_h, mode_x = self.phonon_operators()
        mode_order = (
            tuple(range(int(self.nsites)))
            if self.mode_order is None
            else tuple(int(mode) for mode in self.mode_order)
        )
        steps = []

        for mode in mode_order:
            block_dim = block_h.shape[0]
            eye_block = np.eye(block_dim, dtype=complex)
            eye_mode = np.eye(self.active_mode_dim, dtype=complex)
            raw_h = (
                np.kron(block_h, eye_mode)
                + np.kron(eye_block, self.omega * mode_h)
                + self.g * np.kron(density_ops[mode], mode_x)
            )
            raw_h = 0.5 * (raw_h + raw_h.T.conj())
            energies, vectors = self._truncate(raw_h)
            product_density_ops = [np.kron(op, eye_mode) for op in density_ops]
            density_ops = [
                vectors.T.conj() @ op @ vectors for op in product_density_ops
            ]
            block_h = np.diag(energies).astype(complex)
            steps.append(
                HolsteinElectronicFirstStep(
                    mode=int(mode),
                    product_dim=raw_h.shape[0],
                    kept=len(energies),
                    lowest_energy=float(energies[0]),
                )
            )

        return HolsteinElectronicFirstResult(
            energies=np.diag(block_h).real[: int(nroots)],
            block_hamiltonian=block_h,
            density_operators=density_ops,
            steps=steps,
        )


@dataclass(frozen=True)
class SpinfulHolsteinElectronicFirstNARG:
    """Spinful Holstein/Holstein-Hubbard NARG with electronic supersite first.

    The first site is the full fixed-``(N_up, N_down)`` electronic Hilbert
    space of a 1D chain.  At half filling the default target is
    ``N_up = N_down = L/2``.  Each subsequent NARG site is one phonon mode.

    The pure Holstein model is obtained with ``hubbard_u=0``.
    """

    nsites: int
    t: float = 0.2
    omega: float = 1.0
    g: float = 1.2
    hubbard_u: float = 0.0
    nphonon: int = 8
    local_dim: int | None = None
    bond_dim: int = 64
    nup: int | None = None
    ndown: int | None = None
    mode_order: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.nsites < 1:
            raise ValueError("nsites must be at least 1.")
        if self.nphonon < 1:
            raise ValueError("nphonon must be at least 1.")
        if self.local_dim is not None:
            if self.local_dim < 1:
                raise ValueError("local_dim must be at least 1.")
            if self.local_dim > self.nphonon:
                raise ValueError("local_dim cannot exceed nphonon.")
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be at least 1.")
        _resolve_spin_sector(self.nsites, self.nup, self.ndown)
        if self.mode_order is not None:
            order = tuple(int(mode) for mode in self.mode_order)
            if sorted(order) != list(range(int(self.nsites))):
                raise ValueError("mode_order must be a permutation of range(nsites).")

    @property
    def target(self) -> tuple[int, int]:
        return _resolve_spin_sector(self.nsites, self.nup, self.ndown)

    @property
    def active_mode_dim(self) -> int:
        return self.nphonon if self.local_dim is None else int(self.local_dim)

    def electronic_basis(self) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
        nup, ndown = self.target
        basis = [
            (up_bits, down_bits)
            for up_bits in _combinations_bits(self.nsites, nup)
            for down_bits in _combinations_bits(self.nsites, ndown)
        ]
        return basis, {state: index for index, state in enumerate(basis)}

    def electronic_hamiltonian(self) -> np.ndarray:
        basis, index = self.electronic_basis()
        hamiltonian = np.zeros((len(basis), len(basis)), dtype=float)
        for col, (up_bits, down_bits) in enumerate(basis):
            for site in range(int(self.nsites)):
                n_up_site = (int(up_bits) >> site) & 1
                n_down_site = (int(down_bits) >> site) & 1
                hamiltonian[col, col] += float(self.hubbard_u) * n_up_site * n_down_site

            bits = _spin_orbital_bits(up_bits, down_bits, self.nsites)
            for site in range(int(self.nsites) - 1):
                for spin_offset in (0, 1):
                    left = 2 * site + spin_offset
                    right = 2 * (site + 1) + spin_offset
                    for create_orbital, annihilate_orbital in (
                        (left, right),
                        (right, left),
                    ):
                        applied = _apply_cdag_c(bits, create_orbital, annihilate_orbital)
                        if applied is None:
                            continue
                        new_bits, sign = applied
                        new_up, new_down = _split_spin_orbital_bits(
                            new_bits, self.nsites
                        )
                        row = index.get((new_up, new_down))
                        if row is not None:
                            hamiltonian[row, col] += -float(self.t) * sign
        return 0.5 * (hamiltonian + hamiltonian.T)

    def electronic_density_operators(self) -> list[np.ndarray]:
        basis, _index = self.electronic_basis()
        operators = []
        for site in range(int(self.nsites)):
            diagonal = [
                ((int(up_bits) >> site) & 1) + ((int(down_bits) >> site) & 1)
                for up_bits, down_bits in basis
            ]
            operators.append(np.diag(np.asarray(diagonal, dtype=float)))
        return operators

    def phonon_operators(self) -> tuple[np.ndarray, np.ndarray]:
        dim = self.active_mode_dim
        b = boson_annihilation(dim)
        bdag = b.T.conj()
        return bdag @ b, b + bdag

    def _truncate(self, hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = min(int(self.bond_dim), hamiltonian.shape[0])
        energies, vectors = eigh(hamiltonian, subset_by_index=(0, keep - 1))
        return energies, vectors

    def run(self, nroots: int = 4) -> SpinfulHolsteinElectronicFirstResult:
        block_h = self.electronic_hamiltonian().astype(complex)
        electronic_dim = block_h.shape[0]
        density_ops = [op.astype(complex) for op in self.electronic_density_operators()]
        mode_h, mode_x = self.phonon_operators()
        mode_order = (
            tuple(range(int(self.nsites)))
            if self.mode_order is None
            else tuple(int(mode) for mode in self.mode_order)
        )
        steps = []

        for mode in mode_order:
            block_dim = block_h.shape[0]
            eye_block = np.eye(block_dim, dtype=complex)
            eye_mode = np.eye(self.active_mode_dim, dtype=complex)
            raw_h = (
                np.kron(block_h, eye_mode)
                + np.kron(eye_block, self.omega * mode_h)
                + self.g * np.kron(density_ops[mode], mode_x)
            )
            raw_h = 0.5 * (raw_h + raw_h.T.conj())
            energies, vectors = self._truncate(raw_h)
            product_density_ops = [np.kron(op, eye_mode) for op in density_ops]
            density_ops = [
                vectors.T.conj() @ op @ vectors for op in product_density_ops
            ]
            block_h = np.diag(energies).astype(complex)
            steps.append(
                HolsteinElectronicFirstStep(
                    mode=int(mode),
                    product_dim=raw_h.shape[0],
                    kept=len(energies),
                    lowest_energy=float(energies[0]),
                )
            )

        return SpinfulHolsteinElectronicFirstResult(
            energies=np.diag(block_h).real[: int(nroots)],
            block_hamiltonian=block_h,
            density_operators=density_ops,
            target=self.target,
            electronic_dim=electronic_dim,
            steps=steps,
        )


@dataclass(frozen=True)
class SpinfulHolsteinAdiabaticElectronicNARG:
    """Half-filled spinful Holstein NARG with conditional electronic states.

    Active phonon modes are represented on a coordinate grid.  By default this
    is a sine DVR in the dimensionless oscillator coordinate ``q`` with box
    boundaries ``[-xmax, xmax]``.  At every active coordinate ``q`` the
    fixed-sector electronic Hamiltonian

    ``H_el(q) = H_el + sqrt(2) g sum_m q_m n_m``

    is diagonalized exactly.  The retained conditional electronic states define
    a moving basis over the active-mode grid; phonon kinetic energy is projected
    through the point-to-point electronic overlaps.  Passing ``mode_transform``
    replaces local active modes by orthonormal collective coordinates with
    density couplings ``sum_i mode_transform[a, i] n_i``.
    """

    nsites: int
    t: float = 0.2
    omega: float = 1.0
    g: float = 1.2
    hubbard_u: float = 0.0
    ngrid: int = 9
    xmax: float = 6.0
    mass: float = 1.0
    phonon_basis: str = "sine_dvr"
    active_modes: tuple[int, ...] | None = None
    mode_transform: np.ndarray | None = None
    mode_strengths: np.ndarray | None = None
    nup: int | None = None
    ndown: int | None = None

    def __post_init__(self):
        if self.nsites < 1:
            raise ValueError("nsites must be at least 1.")
        if self.ngrid < 1:
            raise ValueError("ngrid must be at least 1.")
        if self.xmax <= 0:
            raise ValueError("xmax must be positive.")
        if self.mass <= 0:
            raise ValueError("mass must be positive.")
        if self._phonon_basis_name() not in {"sine_dvr", "finite_difference"}:
            raise ValueError("phonon_basis must be sine_dvr or finite_difference.")
        if self._phonon_basis_name() == "finite_difference" and self.ngrid < 3:
            raise ValueError("finite_difference phonon_basis needs ngrid at least 3.")
        _resolve_spin_sector(self.nsites, self.nup, self.ndown)
        self._active_modes_tuple()
        self._mode_strengths_array()

    @property
    def target(self) -> tuple[int, int]:
        return _resolve_spin_sector(self.nsites, self.nup, self.ndown)

    def _phonon_basis_name(self) -> str:
        name = str(self.phonon_basis).lower().replace("-", "_")
        aliases = {
            "sine": "sine_dvr",
            "sinedvr": "sine_dvr",
            "sine_dvr": "sine_dvr",
            "box_dvr": "sine_dvr",
            "dvr": "sine_dvr",
            "coordinate": "finite_difference",
            "fd": "finite_difference",
            "finite_difference": "finite_difference",
            "finite_diff": "finite_difference",
        }
        return aliases.get(name, name)

    def _coordinate_coupling(self) -> float:
        if self._phonon_basis_name() == "sine_dvr":
            return float(np.sqrt(2.0) * self.g)
        return float(self.g)

    def _active_modes_tuple(self) -> tuple[int, ...]:
        transform = self._mode_transform_matrix()
        if transform is not None:
            if self.active_modes is not None:
                raise ValueError("active_modes and mode_transform are mutually exclusive.")
            return tuple(range(transform.shape[0]))
        if self.active_modes is None:
            modes = tuple(range(int(self.nsites)))
        else:
            modes = tuple(int(mode) for mode in self.active_modes)
        if not modes:
            raise ValueError("active_modes must contain at least one mode.")
        if len(set(modes)) != len(modes):
            raise ValueError("active_modes contains duplicate indices.")
        if min(modes) < 0 or max(modes) >= int(self.nsites):
            raise ValueError("active_modes contains an out-of-range site.")
        return modes

    def _mode_transform_matrix(self) -> np.ndarray | None:
        if self.mode_transform is None:
            return None
        transform = np.asarray(self.mode_transform, dtype=float)
        if transform.ndim != 2:
            raise ValueError("mode_transform must be a two-dimensional array.")
        if transform.shape[0] < 1:
            raise ValueError("mode_transform must contain at least one collective mode.")
        if transform.shape[1] != int(self.nsites):
            raise ValueError("mode_transform must have one column per Holstein site.")
        row_overlap = transform @ transform.T
        if not np.allclose(row_overlap, np.eye(transform.shape[0]), atol=1e-8):
            raise ValueError("mode_transform rows must be orthonormal.")
        return transform

    def _mode_strengths_array(self) -> np.ndarray | None:
        if self.mode_strengths is None:
            return None
        strengths = np.asarray(self.mode_strengths, dtype=float).reshape(-1)
        if strengths.shape[0] != len(self._active_modes_tuple()):
            raise ValueError("mode_strengths length must match the number of active modes.")
        return strengths

    def active_mode_site_weights(self) -> np.ndarray:
        """Return site-density weights for each active local/collective mode."""
        transform = self._mode_transform_matrix()
        if transform is not None:
            return transform

        weights = np.zeros((len(self._active_modes_tuple()), int(self.nsites)), dtype=float)
        for row, mode in enumerate(self._active_modes_tuple()):
            weights[row, mode] = 1.0
        return weights

    def active_density_operators(self) -> list[np.ndarray]:
        """Return density coupling operators for local or collective modes."""
        densities = self.electronic_density_operators()
        operators = []
        for row in self.active_mode_site_weights():
            operator = np.zeros_like(densities[0], dtype=float)
            for weight, density in zip(row, densities):
                operator = operator + float(weight) * density
            operators.append(0.5 * (operator + operator.T))
        return operators

    @property
    def electronic_dim(self) -> int:
        nup, ndown = self.target
        return comb(int(self.nsites), int(nup)) * comb(int(self.nsites), int(ndown))

    @property
    def phonon_dim(self) -> int:
        return int(self.ngrid) ** len(self._active_modes_tuple())

    @property
    def dim(self) -> int:
        return self.phonon_dim * self.electronic_dim

    def letta_product_dims(
        self,
        order: str = "mode-first",
        *,
        electronic_dim: int | None = None,
    ) -> tuple[int, ...]:
        """Return LETTA product dimensions for the requested site order."""
        order = _normalize_letta_holstein_order(order)
        if electronic_dim is None:
            electronic_dim = self.electronic_dim
        electronic_dim = int(electronic_dim)
        if electronic_dim < 1:
            raise ValueError("electronic_dim must be positive.")
        mode_dims = (int(self.ngrid),) * len(self._active_modes_tuple())
        if order == "mode-first":
            return mode_dims + (electronic_dim,)
        return (electronic_dim,) + mode_dims

    def letta_mpo(
        self,
        order: str = "mode-first",
        *,
        electronic_hamiltonian: np.ndarray | None = None,
        density_operators: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
        electronic_basis: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Return the exact product-basis MPO for LETTA sweeps.

        The default physical order is all active phonon DVR coordinates followed
        by one electronic supersite.  Passing ``order="electronic-first"`` puts
        the electronic supersite first, which matches the direct sequential NARG
        tensor export.  Star couplings between the electronic supersite and each
        active mode are carried by separate MPO channels.  Passing projected
        electronic operators gives an MPO in a compressed electronic basis.
        """
        order = _normalize_letta_holstein_order(order)
        nactive = len(self._active_modes_tuple())
        ngrid = int(self.ngrid)
        bond = nactive + 2
        start = 0
        done = bond - 1

        if electronic_basis is not None:
            electronic_basis = np.asarray(electronic_basis, dtype=float)
            if electronic_basis.ndim != 2 or electronic_basis.shape[0] != self.electronic_dim:
                raise ValueError("electronic_basis must have shape (electronic_dim, projected_dim).")

        if electronic_hamiltonian is None:
            h_electronic = self.electronic_hamiltonian()
            if electronic_basis is not None:
                h_electronic = electronic_basis.T @ h_electronic @ electronic_basis
        else:
            h_electronic = np.asarray(electronic_hamiltonian)
            if h_electronic.ndim != 2 or h_electronic.shape[0] != h_electronic.shape[1]:
                raise ValueError("electronic_hamiltonian must be square.")

        if density_operators is None:
            if electronic_basis is None:
                densities = self.active_density_operators()
            else:
                densities = []
                for weights in self.active_mode_site_weights():
                    diagonal = np.zeros(self.electronic_dim, dtype=float)
                    for weight, density in zip(weights, self.electronic_density_diagonals()):
                        diagonal = diagonal + float(weight) * density
                    projected = electronic_basis.T @ (diagonal[:, None] * electronic_basis)
                    densities.append(0.5 * (projected + projected.T))
        else:
            densities = [np.asarray(operator) for operator in density_operators]
            if len(densities) != nactive:
                raise ValueError("density_operators must contain one operator per active mode.")

        electronic_dim = h_electronic.shape[0]
        for operator in densities:
            if operator.shape != (electronic_dim, electronic_dim):
                raise ValueError("density operator dimensions must match electronic_hamiltonian.")

        eye_mode = np.eye(ngrid)
        q_mode = np.diag(self.grid())
        h_mode = self.phonon_kinetic_1d() + np.diag(self.phonon_potential_1d())
        coupling = self._coordinate_coupling()
        dtype = np.result_type(
            h_mode,
            q_mode,
            h_electronic,
            *densities,
        )

        if order == "electronic-first":
            electronic = np.zeros((bond, bond, electronic_dim, electronic_dim), dtype=dtype)
            electronic[start, start] = np.eye(electronic_dim)
            electronic[start, done] = h_electronic
            for mode_index, density in enumerate(densities):
                electronic[start, 1 + mode_index] = coupling * density
            mpo = [electronic[start : start + 1]]

            for mode_index in range(nactive):
                site = np.zeros((bond, bond, ngrid, ngrid), dtype=dtype)
                site[start, start] = eye_mode
                site[done, done] = eye_mode
                site[start, done] = h_mode
                for carrier in range(nactive):
                    site[1 + carrier, 1 + carrier] = eye_mode
                site[1 + mode_index, done] = q_mode
                if mode_index == nactive - 1:
                    site = site[:, done : done + 1]
                mpo.append(site)
            return mpo

        mpo = []
        for mode_index in range(nactive):
            site = np.zeros((bond, bond, ngrid, ngrid), dtype=dtype)
            site[start, start] = eye_mode
            site[done, done] = eye_mode
            site[start, done] = h_mode
            site[start, 1 + mode_index] = coupling * q_mode
            for carrier in range(nactive):
                site[1 + carrier, 1 + carrier] = eye_mode
            if mode_index == 0:
                site = site[start : start + 1]
            mpo.append(site)

        terminal = np.zeros((bond, 1, electronic_dim, electronic_dim), dtype=dtype)
        terminal[start, 0] = h_electronic
        terminal[done, 0] = np.eye(electronic_dim)
        for mode_index, density in enumerate(densities):
            terminal[1 + mode_index, 0] = density
        mpo.append(terminal)
        return mpo

    def grid(self) -> np.ndarray:
        if self._phonon_basis_name() == "sine_dvr":
            return sine_dvr_grid(int(self.ngrid), float(self.xmax))
        return np.linspace(-float(self.xmax), float(self.xmax), int(self.ngrid))

    def coordinate_mesh(self) -> np.ndarray:
        axes = np.meshgrid(
            *([self.grid()] * len(self._active_modes_tuple())),
            indexing="ij",
        )
        return np.stack([axis.reshape(-1) for axis in axes], axis=1)

    def phonon_potential(self) -> np.ndarray:
        coords = self.coordinate_mesh()
        if self._phonon_basis_name() == "sine_dvr":
            nactive = len(self._active_modes_tuple())
            return float(self.omega) * (0.5 * np.sum(coords * coords, axis=1) - 0.5 * nactive)
        return 0.5 * float(self.mass) * self.omega**2 * np.sum(coords * coords, axis=1)

    def phonon_kinetic(self) -> np.ndarray:
        if self._phonon_basis_name() == "sine_dvr":
            kinetic_1d = float(self.omega) * sine_dvr_kinetic(
                int(self.ngrid),
                float(self.xmax),
            )
        else:
            kinetic_1d = finite_difference_kinetic(self.grid(), mass=self.mass)
        eye = np.eye(int(self.ngrid))
        kinetic = np.array([[0.0]])
        for mode_index in range(len(self._active_modes_tuple())):
            factors = [
                kinetic_1d if axis == mode_index else eye
                for axis in range(len(self._active_modes_tuple()))
            ]
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            kinetic = term if kinetic.shape == (1, 1) else kinetic + term
        return kinetic

    def phonon_potential_1d(self) -> np.ndarray:
        grid = self.grid()
        if self._phonon_basis_name() == "sine_dvr":
            return float(self.omega) * (0.5 * grid * grid - 0.5)
        return 0.5 * float(self.mass) * self.omega**2 * grid * grid

    def phonon_kinetic_1d(self) -> np.ndarray:
        if self._phonon_basis_name() == "sine_dvr":
            return float(self.omega) * sine_dvr_kinetic(
                int(self.ngrid),
                float(self.xmax),
            )
        return finite_difference_kinetic(self.grid(), mass=self.mass)

    def _electronic_helper(self) -> SpinfulHolsteinElectronicFirstNARG:
        return SpinfulHolsteinElectronicFirstNARG(
            self.nsites,
            t=self.t,
            omega=self.omega,
            g=self.g,
            hubbard_u=self.hubbard_u,
            nphonon=1,
            nup=self.nup,
            ndown=self.ndown,
        )

    def electronic_hamiltonian(self) -> np.ndarray:
        return self._electronic_helper().electronic_hamiltonian()

    def electronic_density_operators(self) -> list[np.ndarray]:
        return self._electronic_helper().electronic_density_operators()

    def electronic_density_diagonals(self) -> list[np.ndarray]:
        basis, _index = self._electronic_helper().electronic_basis()
        diagonals = []
        for site in range(int(self.nsites)):
            diagonals.append(
                np.asarray(
                    [
                        ((int(up_bits) >> site) & 1)
                        + ((int(down_bits) >> site) & 1)
                        for up_bits, down_bits in basis
                    ],
                    dtype=float,
                )
            )
        return diagonals

    def projected_active_density_operators(self, vectors: np.ndarray) -> list[np.ndarray]:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != self.electronic_dim:
            raise ValueError("vectors must have shape (electronic_dim, kept_states).")

        density_diagonals = self.electronic_density_diagonals()
        operators = []
        for weights in self.active_mode_site_weights():
            diagonal = np.zeros(self.electronic_dim, dtype=float)
            for weight, density in zip(weights, density_diagonals):
                diagonal = diagonal + float(weight) * density
            operator = vectors.T @ (diagonal[:, None] * vectors)
            operators.append(0.5 * (operator + operator.T))
        return operators

    def electronic_hamiltonian_at(self, coordinates: np.ndarray) -> np.ndarray:
        hamiltonian = self.electronic_hamiltonian().astype(float)
        densities = self.active_density_operators()
        coordinate = np.asarray(coordinates, dtype=float).reshape(-1)
        if coordinate.shape[0] != len(densities):
            raise ValueError("coordinates length must match the number of active modes.")
        coupling = self._coordinate_coupling()
        for value, density in zip(coordinate, densities):
            hamiltonian = hamiltonian + coupling * float(value) * density
        return 0.5 * (hamiltonian + hamiltonian.T)

    def one_body_hamiltonian_at(self, coordinates: np.ndarray) -> np.ndarray:
        """Return the conditional one-electron Hamiltonian for ``hubbard_u=0``."""
        coordinate = np.asarray(coordinates, dtype=float).reshape(-1)
        weights = self.active_mode_site_weights()
        if coordinate.shape[0] != weights.shape[0]:
            raise ValueError("coordinates length must match the number of active modes.")

        potential = self._coordinate_coupling() * (coordinate @ weights)
        hamiltonian = np.diag(potential.astype(float))
        for site in range(int(self.nsites) - 1):
            hamiltonian[site, site + 1] = -float(self.t)
            hamiltonian[site + 1, site] = -float(self.t)
        return 0.5 * (hamiltonian + hamiltonian.T)

    def _uses_quadratic_electronic_solver(self) -> bool:
        return abs(float(self.hubbard_u)) < 1e-14

    def _site_orbital_combinations(self, nelec: int) -> list[tuple[int, ...]]:
        return [tuple(combo) for combo in combinations(range(int(self.nsites)), int(nelec))]

    def _slater_amplitude_matrix(
        self,
        orbitals: np.ndarray,
        site_configs: list[int],
        orbital_configs: list[tuple[int, ...]],
    ) -> np.ndarray:
        orbitals = np.asarray(orbitals, dtype=float)
        out = np.empty((len(site_configs), len(orbital_configs)), dtype=float)
        if not orbital_configs or len(orbital_configs[0]) == 0:
            out.fill(1.0)
            return out

        for row, bits in enumerate(site_configs):
            sites = [site for site in range(int(self.nsites)) if (int(bits) >> site) & 1]
            for col, occupied_orbitals in enumerate(orbital_configs):
                out[row, col] = np.linalg.det(
                    orbitals[np.ix_(sites, occupied_orbitals)]
                )
        return out

    def _spin_interleaving_signs(
        self,
        up_configs: list[int],
        down_configs: list[int],
    ) -> np.ndarray:
        signs = np.ones((len(up_configs), len(down_configs)), dtype=float)
        for up_index, up_bits in enumerate(up_configs):
            for down_index, down_bits in enumerate(down_configs):
                inversions = 0
                for up_site in range(int(self.nsites)):
                    if (int(up_bits) >> up_site) & 1:
                        inversions += _bit_count(int(down_bits) & ((1 << up_site) - 1))
                signs[up_index, down_index] = -1.0 if inversions % 2 else 1.0
        return signs

    def _quadratic_slater_eigensystem(
        self,
        one_body_hamiltonian: np.ndarray,
        nstates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        nup, ndown = self.target
        up_configs = _combinations_bits(self.nsites, nup)
        down_configs = _combinations_bits(self.nsites, ndown)
        up_orbital_configs = self._site_orbital_combinations(nup)
        down_orbital_configs = self._site_orbital_combinations(ndown)
        signs = self._spin_interleaving_signs(up_configs, down_configs)
        edim = len(up_configs) * len(down_configs)
        if nstates is None:
            keep = edim
        else:
            keep = min(max(int(nstates), 1), edim)

        orbital_energies, orbitals = eigh(one_body_hamiltonian)
        up_amplitudes = self._slater_amplitude_matrix(
            orbitals,
            up_configs,
            up_orbital_configs,
        )
        down_amplitudes = self._slater_amplitude_matrix(
            orbitals,
            down_configs,
            down_orbital_configs,
        )

        state_records = []
        for up_index, up_orbitals in enumerate(up_orbital_configs):
            up_energy = float(np.sum(orbital_energies[list(up_orbitals)]))
            for down_index, down_orbitals in enumerate(down_orbital_configs):
                energy = up_energy + float(np.sum(orbital_energies[list(down_orbitals)]))
                state_records.append((energy, up_index, down_index))
        state_records.sort(key=lambda item: item[0])
        state_records = state_records[:keep]

        energies = np.empty(keep, dtype=float)
        vectors = np.empty((edim, keep), dtype=float)
        for column, (energy, up_index, down_index) in enumerate(state_records):
            energies[column] = energy
            amplitudes = (
                signs
                * up_amplitudes[:, up_index, None]
                * down_amplitudes[None, :, down_index]
            )
            vectors[:, column] = amplitudes.reshape(-1)
        return energies, vectors

    def electronic_eigensystem(
        self,
        nstates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return electronic eigenstates in the fixed spin sector."""
        if self._uses_quadratic_electronic_solver():
            zero = np.zeros(len(self._active_modes_tuple()), dtype=float)
            return self._quadratic_slater_eigensystem(
                self.one_body_hamiltonian_at(zero),
                nstates=nstates,
            )

        hamiltonian = self.electronic_hamiltonian()
        if nstates is None or int(nstates) >= hamiltonian.shape[0]:
            return eigh(hamiltonian)
        nstates = max(int(nstates), 1)
        return eigh(hamiltonian, subset_by_index=(0, nstates - 1))

    def quadratic_conditional_states(
        self,
        nstates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conditional many-electron states from one-body Slater determinants."""
        if not self._uses_quadratic_electronic_solver():
            raise ValueError("quadratic_conditional_states requires hubbard_u=0.")

        coords = self.coordinate_mesh()
        edim = self.electronic_dim
        if nstates is None:
            keep = edim
        else:
            keep = min(max(int(nstates), 1), edim)
        energies = np.empty((coords.shape[0], keep), dtype=float)
        vectors = np.empty((coords.shape[0], edim, keep), dtype=float)

        for point, coordinate in enumerate(coords):
            point_energies, point_vectors = self._quadratic_slater_eigensystem(
                self.one_body_hamiltonian_at(coordinate),
                nstates=keep,
            )
            energies[point] = point_energies
            vectors[point] = point_vectors

        self._align_conditional_state_signs(vectors)
        return energies, vectors

    def density_response_mode_transform(
        self,
        *,
        nlow: int | None = None,
        center: bool = True,
    ) -> SpinfulHolsteinModeTransform:
        """Order collective phonon modes by low-energy density-response strength.

        The Gram matrix is a Hilbert-Schmidt overlap of the electronic density
        operators projected into the lowest ``nlow`` eigenstates of the purely
        electronic fixed-sector Hamiltonian.  Its leading eigenvectors define
        orthonormal collective phonon coordinates with descending coupling
        strength.
        """
        if nlow is None:
            nlow = self.electronic_dim
        nlow = int(nlow)
        if nlow < 1:
            raise ValueError("nlow must be at least 1.")
        nlow = min(nlow, self.electronic_dim)
        electronic_energies, low_vectors = self.electronic_eigensystem(nstates=nlow)
        nlow = low_vectors.shape[1]
        projected = []
        identity = np.eye(nlow)
        for density_diagonal in self.electronic_density_diagonals():
            operator = low_vectors.T.conj() @ (density_diagonal[:, None] * low_vectors)
            operator = 0.5 * (operator + operator.T.conj())
            if center:
                operator = operator - (np.trace(operator).real / nlow) * identity
            projected.append(operator)

        gram = np.empty((int(self.nsites), int(self.nsites)), dtype=float)
        for i, left in enumerate(projected):
            for j, right in enumerate(projected):
                gram[i, j] = float(np.vdot(left, right).real / nlow)
        gram = 0.5 * (gram + gram.T)

        values, vectors = eigh(gram)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0.0)
        values[values < 1e-14] = 0.0
        transform = vectors[:, order].T
        for row in transform:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0:
                row *= -1.0
        strengths = abs(self._coordinate_coupling()) * np.sqrt(values)
        return SpinfulHolsteinModeTransform(
            transform=transform,
            strengths=strengths,
            gram=gram,
            electronic_energies=electronic_energies[:nlow],
            nlow=nlow,
            centered=bool(center),
        )

    def conditional_states(self) -> tuple[np.ndarray, np.ndarray]:
        if self._uses_quadratic_electronic_solver():
            return self.quadratic_conditional_states()

        coords = self.coordinate_mesh()
        edim = self.electronic_dim
        energies = np.empty((coords.shape[0], edim), dtype=float)
        vectors = np.empty((coords.shape[0], edim, edim), dtype=float)
        for point, coordinate in enumerate(coords):
            local_energies, local_vectors = eigh(self.electronic_hamiltonian_at(coordinate))
            energies[point] = local_energies
            vectors[point] = local_vectors
        self._align_conditional_state_signs(vectors)
        return energies, vectors

    def _align_conditional_state_signs(self, vectors: np.ndarray) -> None:
        for point in range(1, vectors.shape[0]):
            reference = point - 1
            overlaps = vectors[reference].T @ vectors[point]
            for state in range(vectors.shape[2]):
                if overlaps[state, state] < 0:
                    vectors[point, :, state] *= -1.0

    def full_hamiltonian(self) -> np.ndarray:
        kinetic = self.phonon_kinetic()
        potential = self.phonon_potential()
        hamiltonian = np.kron(
            kinetic + np.diag(potential),
            np.eye(self.electronic_dim),
        )
        for point, coordinate in enumerate(self.coordinate_mesh()):
            start = point * self.electronic_dim
            stop = start + self.electronic_dim
            hamiltonian[start:stop, start:stop] += self.electronic_hamiltonian_at(
                coordinate
            )
        return 0.5 * (hamiltonian + hamiltonian.T)

    def effective_hamiltonian(
        self,
        nstates_per_point: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nstates_per_point = int(nstates_per_point)
        if nstates_per_point < 1 or nstates_per_point > self.electronic_dim:
            raise ValueError("nstates_per_point must be between 1 and electronic_dim.")

        conditional_energies, conditional_vectors = self.conditional_states()
        active_vectors = conditional_vectors[:, :, :nstates_per_point]
        kinetic = self.phonon_kinetic()
        potential = self.phonon_potential()
        hamiltonian = np.einsum(
            "pq,pia,qib->paqb",
            kinetic,
            active_vectors,
            active_vectors,
            optimize=True,
        )
        hamiltonian = hamiltonian.reshape(
            self.phonon_dim * nstates_per_point,
            self.phonon_dim * nstates_per_point,
        )
        diagonal = (
            potential[:, None]
            + conditional_energies[:, :nstates_per_point]
        ).reshape(-1)
        hamiltonian += np.diag(diagonal)
        return (
            0.5 * (hamiltonian + hamiltonian.T.conj()),
            conditional_energies,
            conditional_vectors,
        )

    def run_sequential(
        self,
        *,
        nstates_per_point: int = 8,
        bond_dim: int = 64,
        initial_electronic_states: int | None = None,
        nroots: int = 4,
        store_narg_state: bool = False,
        narg_electronic_basis: str = "full",
    ) -> SpinfulHolsteinSequentialAdiabaticResult:
        """Add active phonon modes one at a time in a conditional basis."""
        nstates_per_point = int(nstates_per_point)
        bond_dim = int(bond_dim)
        if nstates_per_point < 1:
            raise ValueError("nstates_per_point must be at least 1.")
        if bond_dim < 1:
            raise ValueError("bond_dim must be at least 1.")
        narg_electronic_basis = str(narg_electronic_basis).lower().replace("_", "-")
        if narg_electronic_basis not in {"full", "initial"}:
            raise ValueError("narg_electronic_basis must be full or initial.")
        if initial_electronic_states is None:
            initial_electronic_states = min(self.electronic_dim, bond_dim)
        initial_electronic_states = min(
            max(int(initial_electronic_states), 1),
            self.electronic_dim,
        )

        electronic_energies, electronic_vectors = self.electronic_eigensystem(
            nstates=initial_electronic_states
        )
        block_h = np.diag(electronic_energies).astype(float)
        density_ops = self.projected_active_density_operators(electronic_vectors)
        narg_tensors = [] if store_narg_state else None
        initial_density_ops = tuple(operator.copy() for operator in density_ops)

        grid = self.grid()
        kinetic = self.phonon_kinetic_1d()
        potential = self.phonon_potential_1d()
        coupling = self._coordinate_coupling()
        steps = []

        for mode_position, mode in enumerate(self._active_modes_tuple()):
            mode = int(mode)
            input_dim = block_h.shape[0]
            conditional_dim = min(nstates_per_point, input_dim)
            conditional_energies = np.empty((int(self.ngrid), conditional_dim), dtype=float)
            conditional_vectors = np.empty(
                (int(self.ngrid), input_dim, conditional_dim),
                dtype=float,
            )
            density = density_ops[mode_position]
            for point, q in enumerate(grid):
                local_h = block_h + coupling * float(q) * density
                local_h = 0.5 * (local_h + local_h.T)
                local_energies, local_vectors = eigh(
                    local_h,
                    subset_by_index=(0, conditional_dim - 1),
                )
                conditional_energies[point] = local_energies
                conditional_vectors[point] = local_vectors
            self._align_conditional_state_signs(conditional_vectors)

            hamiltonian = np.einsum(
                "pq,pia,qib->paqb",
                kinetic,
                conditional_vectors,
                conditional_vectors,
                optimize=True,
            )
            hamiltonian = hamiltonian.reshape(
                int(self.ngrid) * conditional_dim,
                int(self.ngrid) * conditional_dim,
            )
            hamiltonian += np.diag(
                (potential[:, None] + conditional_energies).reshape(-1)
            )
            hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)

            keep = min(bond_dim, hamiltonian.shape[0])
            energies, vectors = eigh(hamiltonian, subset_by_index=(0, keep - 1))

            if narg_tensors is not None:
                local_vectors = vectors.reshape(int(self.ngrid), conditional_dim, keep)
                growth = np.einsum(
                    "pia,pab->pib",
                    conditional_vectors,
                    local_vectors,
                    optimize=True,
                )
                if mode_position == 0:
                    if narg_electronic_basis == "initial":
                        tensor = np.moveaxis(growth, 0, 2)
                    else:
                        tensor = np.einsum(
                            "ia,pab->ibp",
                            electronic_vectors,
                            growth,
                            optimize=True,
                        )
                else:
                    tensor = np.empty(
                        (int(self.ngrid) * input_dim, keep, int(self.ngrid)),
                        dtype=growth.dtype,
                    )
                    block = np.moveaxis(growth, 0, 2)
                    for previous_point in range(int(self.ngrid)):
                        start = previous_point * input_dim
                        stop = start + input_dim
                        tensor[start:stop] = block
                narg_tensors.append(tensor)

            new_density_ops = []
            for old_density in density_ops:
                projected = np.zeros_like(hamiltonian)
                for point in range(int(self.ngrid)):
                    start = point * conditional_dim
                    stop = start + conditional_dim
                    local_projected = (
                        conditional_vectors[point].T
                        @ old_density
                        @ conditional_vectors[point]
                    )
                    projected[start:stop, start:stop] = local_projected
                new_density = vectors.T @ projected @ vectors
                new_density_ops.append(0.5 * (new_density + new_density.T))
            density_ops = new_density_ops
            block_h = np.diag(energies)
            steps.append(
                SpinfulHolsteinSequentialAdiabaticStep(
                    mode=mode,
                    input_dim=input_dim,
                    grid_dim=int(self.ngrid),
                    conditional_dim=conditional_dim,
                    hamiltonian_dim=hamiltonian.shape[0],
                    kept=keep,
                    lowest_energy=float(energies[0]),
                )
            )

        final_energies = np.diag(block_h).real[: int(nroots)]
        narg_coefficients = None
        narg_dims = None
        if narg_tensors is not None:
            final_dim = block_h.shape[0]
            nstored_roots = final_energies.shape[0]
            narg_coefficients = np.zeros(
                (int(self.ngrid) * final_dim, nstored_roots),
                dtype=float,
            )
            for root in range(nstored_roots):
                for point in range(int(self.ngrid)):
                    narg_coefficients[point * final_dim + root, root] = 1.0
            exported_electronic_dim = (
                electronic_vectors.shape[1]
                if narg_electronic_basis == "initial"
                else self.electronic_dim
            )
            narg_dims = self.letta_product_dims(
                order="electronic-first",
                electronic_dim=exported_electronic_dim,
            )

        return SpinfulHolsteinSequentialAdiabaticResult(
            energies=final_energies,
            block_hamiltonian=block_h,
            density_operators=density_ops,
            target=self.target,
            electronic_dim=self.electronic_dim,
            steps=steps,
            mode_transform=self._mode_transform_matrix(),
            mode_strengths=self._mode_strengths_array(),
            narg_tensors=None if narg_tensors is None else tuple(narg_tensors),
            narg_coefficients=narg_coefficients,
            narg_dims=narg_dims,
            narg_electronic_basis=(
                electronic_vectors.copy()
                if narg_tensors is not None and narg_electronic_basis == "initial"
                else None
            ),
            narg_electronic_hamiltonian=(
                np.diag(electronic_energies).copy()
                if narg_tensors is not None and narg_electronic_basis == "initial"
                else None
            ),
            narg_density_operators=(
                initial_density_ops
                if narg_tensors is not None and narg_electronic_basis == "initial"
                else None
            ),
        )

    def exact(self, nroots: int = 4) -> tuple[np.ndarray, np.ndarray]:
        return _lowest_eigensystem(self.full_hamiltonian(), nroots)

    def run(
        self,
        nstates_per_point: int = 1,
        nroots: int = 4,
    ) -> SpinfulHolsteinAdiabaticElectronicResult:
        hamiltonian, conditional_energies, conditional_vectors = self.effective_hamiltonian(
            nstates_per_point=nstates_per_point
        )
        energies, vectors = _lowest_eigensystem(hamiltonian, nroots)
        return SpinfulHolsteinAdiabaticElectronicResult(
            energies=energies,
            vectors=vectors,
            conditional_energies=conditional_energies,
            conditional_vectors=conditional_vectors[:, :, : int(nstates_per_point)],
            hamiltonian=hamiltonian,
            nstates_per_point=int(nstates_per_point),
            active_modes=self._active_modes_tuple(),
            target=self.target,
            mode_transform=self._mode_transform_matrix(),
            mode_strengths=self._mode_strengths_array(),
        )

    def reconstruct_wavefunction(
        self,
        coefficients: np.ndarray,
        conditional_vectors: np.ndarray,
    ) -> np.ndarray:
        conditional_vectors = np.asarray(conditional_vectors)
        nstates_per_point = conditional_vectors.shape[2]
        coefficients = np.asarray(coefficients).reshape(
            self.phonon_dim, nstates_per_point
        )
        return np.einsum(
            "pia,pa->pi",
            conditional_vectors,
            coefficients,
            optimize=True,
        )


__all__ = [
    "HolsteinDimer",
    "HolsteinDimerConditionalResult",
    "HolsteinAdiabaticNARGResult",
    "HolsteinAdiabaticStep",
    "HolsteinChainBlock",
    "HolsteinChainAdiabaticNARG",
    "HolsteinChainCouplingNARG",
    "HolsteinDressedSite",
    "HolsteinElectronicFirstNARG",
    "HolsteinElectronicFirstResult",
    "HolsteinElectronicFirstStep",
    "HolsteinChainNARG",
    "HolsteinChainNARGResult",
    "HolsteinDimerCoordinateNARG",
    "HolsteinDimerReport",
    "RankDFrameReport",
    "SpinfulHHBlock",
    "SpinfulHHBipolaronDiagnostics",
    "SpinfulHHConditioningStep",
    "SpinfulHHCouplingNARGResult",
    "SpinfulHHDressedSite",
    "SpinfulHHDressedPair",
    "SpinfulHolsteinAdiabaticElectronicNARG",
    "SpinfulHolsteinAdiabaticElectronicResult",
    "SpinfulHolsteinElectronicFirstNARG",
    "SpinfulHolsteinElectronicFirstResult",
    "SpinfulHolsteinModeTransform",
    "SpinfulHolsteinSequentialAdiabaticResult",
    "SpinfulHolsteinSequentialAdiabaticStep",
    "SpinfulHHNARGResult",
    "SpinfulHolsteinHubbardCouplingNARG",
    "SpinfulHolsteinHubbardNARG",
    "SpinfulHolsteinHubbardTwoSiteNARG",
    "boson_annihilation",
    "conditional_rank1_factor",
    "discarded_weight",
    "finite_difference_kinetic",
    "holstein_chain_exact_energies",
    "holstein_chain_exact_hamiltonian",
    "reconstruct_conditional_factor",
    "schmidt_spectrum",
    "sine_dvr_grid",
    "sine_dvr_kinetic",
    "spinful_holstein_hubbard_exact_energies",
    "spinful_holstein_hubbard_exact_hamiltonian",
    "spinful_hh_bipolaron_diagnostics",
    "spinful_hh_pair_binding_energy",
    "truncate_schmidt_state",
]
