"""NARG for spin-boson Wilson chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh


def spin_operators():
    """Return I, X, Y, Z spin-1/2 Pauli matrices."""
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return identity, x, y, z


def boson_operators(dim: int):
    """Return I, b, b^dagger, n for a truncated boson mode."""
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive.")
    identity = np.eye(dim, dtype=complex)
    b = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        b[n - 1, n] = np.sqrt(n)
    bdag = b.T.conj()
    number = bdag @ b
    return identity, b, bdag, number


def boson_dvr_operators(dim: int):
    """Return boson operators in the truncated oscillator coordinate-DVR basis.

    The DVR grid is obtained by diagonalizing ``x=(b+b^dagger)/sqrt(2)`` in a
    finite Fock space.  This is a unitary basis change inside the same cutoff,
    so dense no-truncation results are identical to the Fock representation.
    """
    identity, b, bdag, number = boson_operators(dim)
    coordinate = (b + bdag) / np.sqrt(2.0)
    grid, vectors = np.linalg.eigh(0.5 * (coordinate + coordinate.T.conj()))
    order = np.argsort(grid)
    transform = vectors[:, order]
    return (
        transform.conj().T @ identity @ transform,
        transform.conj().T @ b @ transform,
        transform.conj().T @ bdag @ transform,
        transform.conj().T @ number @ transform,
        np.asarray(grid[order], dtype=float),
        transform,
    )


def boson_displaced_dvr_operators(
    dim: int,
    displacement: float,
    *,
    parent_dim: int | None = None,
    symmetric: bool = True,
):
    """Return oscillator operators in a displaced coordinate-DVR subspace.

    The local variational subspace is built in a larger parent Fock space from
    displaced oscillator states.  With ``symmetric=True`` both ``+alpha`` and
    ``-alpha`` displaced branches are included before compressing back to
    ``dim`` states, which is useful for the unbiased spin-boson model.
    """
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive.")
    parent_dim = max(dim, int(parent_dim or max(dim + 4, 2 * dim)))
    _, b_parent, bdag_parent, number_parent = boson_operators(parent_dim)
    generator = bdag_parent - b_parent

    shifts = (float(displacement),)
    if symmetric and abs(displacement) > 1e-14:
        shifts = (float(displacement), -float(displacement))

    candidates = []
    for shift in shifts:
        displaced = expm(shift * generator)
        candidates.append(displaced[:, :dim])
    candidate_matrix = np.concatenate(candidates, axis=1)
    left, _, _ = np.linalg.svd(candidate_matrix, full_matrices=False)
    subspace = left[:, :dim]

    identity = subspace.conj().T @ np.eye(parent_dim, dtype=complex) @ subspace
    b = subspace.conj().T @ b_parent @ subspace
    bdag = subspace.conj().T @ bdag_parent @ subspace
    number = subspace.conj().T @ number_parent @ subspace

    coordinate = (b + bdag) / np.sqrt(2.0)
    grid, rotate = np.linalg.eigh(0.5 * (coordinate + coordinate.T.conj()))
    order = np.argsort(grid)
    rotate = rotate[:, order]
    return (
        rotate.conj().T @ identity @ rotate,
        rotate.conj().T @ b @ rotate,
        rotate.conj().T @ bdag @ rotate,
        rotate.conj().T @ number @ rotate,
        np.asarray(grid[order], dtype=float),
        subspace @ rotate,
    )


def sine_dvr_boson_operators(
    dim: int,
    *,
    qmax: float = 8.0,
    center: float = 0.0,
):
    """Return oscillator operators on a sine DVR coordinate grid.

    Coordinates are dimensionless oscillator coordinates ``q`` where
    ``b=(q+i p)/sqrt(2)`` and ``p=-i d/dq``.  The grid uses interior
    particle-in-a-box points on ``[center-qmax, center+qmax]``.
    """
    dim = int(dim)
    qmax = float(qmax)
    center = float(center)
    if dim < 1:
        raise ValueError("dim must be positive.")
    if qmax <= 0.0:
        raise ValueError("qmax must be positive.")

    indices = np.arange(1, dim + 1, dtype=float)
    length = 2.0 * qmax
    grid = center - qmax + length * indices / (dim + 1)
    transform = np.sqrt(2.0 / (dim + 1)) * np.sin(
        np.pi * np.outer(indices, indices) / (dim + 1)
    )

    modes = indices
    kinetic_eigs = 0.5 * (np.pi * modes / length) ** 2
    kinetic = (transform * kinetic_eigs) @ transform.T

    derivative = np.zeros((dim, dim), dtype=float)
    for m in range(1, dim + 1):
        for n in range(1, dim + 1):
            if m == n:
                continue
            value = (1.0 - (-1.0) ** (m + n)) / (m + n)
            value += (1.0 - (-1.0) ** (m - n)) / (m - n)
            derivative[m - 1, n - 1] = n * value / length
    momentum_fbr = -1.0j * derivative
    momentum = transform @ momentum_fbr @ transform.T

    identity = np.eye(dim, dtype=complex)
    coordinate = np.diag(grid.astype(complex))
    b = (coordinate + 1.0j * momentum) / np.sqrt(2.0)
    bdag = b.T.conj()
    number = bdag @ b
    # Keep the local harmonic oscillator Hamiltonian tied to the sine-DVR
    # kinetic operator rather than to p@p, whose finite-basis derivative is
    # less accurate near Dirichlet boundaries.
    oscillator = kinetic + 0.5 * coordinate @ coordinate - 0.5 * identity
    oscillator = 0.5 * (oscillator + oscillator.T.conj())
    return identity, b, bdag, oscillator, grid, kinetic, momentum


def local_boson_operators(
    dim: int,
    *,
    basis: str = "dvr",
    displacement: float = 0.0,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
):
    """Return local boson operators for ``basis='dvr'`` or ``basis='fock'``."""
    key = str(basis).lower()
    if key in {"fock", "number", "occupation"}:
        identity, b, bdag, number = boson_operators(dim)
        return identity, b, bdag, number
    if key in {
        "dvr",
        "gh-dvr",
        "gh_dvr",
        "gauss-hermite-dvr",
        "gauss_hermite_dvr",
        "coordinate",
        "oscillator-dvr",
        "oscillator_dvr",
    }:
        if abs(displacement) > 0.0:
            identity, b, bdag, number, _, _ = boson_displaced_dvr_operators(
                dim,
                displacement,
                parent_dim=parent_dim,
                symmetric=symmetric_displacement,
            )
        else:
            identity, b, bdag, number, _, _ = boson_dvr_operators(dim)
        return identity, b, bdag, number
    if key in {"displaced-dvr", "displaced_dvr", "adaptive-dvr", "adaptive_dvr"}:
        identity, b, bdag, number, _, _ = boson_displaced_dvr_operators(
            dim,
            displacement,
            parent_dim=parent_dim,
            symmetric=symmetric_displacement,
        )
        return identity, b, bdag, number
    if key in {"sine-dvr", "sine_dvr", "sinedvr"}:
        q_shift = np.sqrt(2.0) * float(displacement)
        if symmetric_displacement:
            center = 0.0
            qmax = float(dvr_qmax) + abs(q_shift)
        else:
            center = q_shift
            qmax = float(dvr_qmax)
        identity, b, bdag, number, _, _, _ = sine_dvr_boson_operators(
            dim,
            qmax=qmax,
            center=center,
        )
        return identity, b, bdag, number
    raise ValueError("basis must be 'dvr'/'gh-dvr', 'sine-dvr', 'displaced-dvr', or 'fock'.")


@dataclass
class SpinBosonWilsonChain:
    """Finite Wilson-chain representation of a spin-boson bath."""

    onsite: np.ndarray
    hopping: np.ndarray
    impurity_coupling: float
    epsilon: float = 0.0
    delta: float = 0.0
    star_frequencies: np.ndarray | None = None
    star_couplings: np.ndarray | None = None
    star_to_chain: np.ndarray | None = None

    def __post_init__(self):
        self.onsite = np.asarray(self.onsite, dtype=float)
        self.hopping = np.asarray(self.hopping, dtype=float)
        self.impurity_coupling = float(self.impurity_coupling)
        self.epsilon = float(self.epsilon)
        self.delta = float(self.delta)
        if self.onsite.ndim != 1:
            raise ValueError("onsite must be one-dimensional.")
        if self.hopping.ndim != 1:
            raise ValueError("hopping must be one-dimensional.")
        if len(self.hopping) != max(0, len(self.onsite) - 1):
            raise ValueError("hopping length must be len(onsite) - 1.")
        if self.star_frequencies is not None:
            self.star_frequencies = np.asarray(self.star_frequencies, dtype=float)
        if self.star_couplings is not None:
            self.star_couplings = np.asarray(self.star_couplings, dtype=float)
        if self.star_to_chain is not None:
            self.star_to_chain = np.asarray(self.star_to_chain, dtype=float)

    @property
    def nmodes(self) -> int:
        return len(self.onsite)

    @classmethod
    def from_sbm(cls, sbm, *, nmodes: int | None = None):
        """Build from an existing ``pyqed.models.impurity.sbm.SBM`` instance."""
        if nmodes is not None:
            sbm.discretize(int(nmodes))
        if getattr(sbm, "onsite", None) is None or getattr(sbm, "hopping", None) is None:
            raise ValueError("SBM object has not been discretized; pass nmodes or call discretize first.")
        return cls(
            onsite=np.asarray(sbm.onsite, dtype=float),
            hopping=np.asarray(sbm.hopping, dtype=float),
            impurity_coupling=float(sbm.t0),
            epsilon=float(getattr(sbm, "epsilon", 0.0)),
            delta=float(getattr(sbm, "delta", 0.0)),
            star_frequencies=getattr(sbm, "xi", None),
            star_couplings=getattr(sbm, "g", None),
        )

    def impurity_hamiltonian(self):
        _, x, _, z = spin_operators()
        return 0.5 * self.epsilon * z - 0.5 * self.delta * x

    def estimate_displacements(self):
        """Classical Wilson-chain oscillator shifts for a localized spin branch."""
        if self.nmodes == 0:
            return np.array([])
        force_matrix = np.diag(self.onsite).astype(float)
        for index, hopping in enumerate(self.hopping):
            force_matrix[index, index + 1] = hopping
            force_matrix[index + 1, index] = hopping
        source = np.zeros(self.nmodes, dtype=float)
        source[0] = 0.5 * self.impurity_coupling
        try:
            shifts = -np.linalg.solve(force_matrix, source)
        except np.linalg.LinAlgError:
            shifts = -np.linalg.lstsq(force_matrix, source, rcond=None)[0]
        return np.abs(shifts)


@dataclass
class SpinBosonWilsonNARGStep:
    """One growth step in spin-boson Wilson-chain NARG."""

    site: int
    product_dim: int
    kept: int
    lowest_energy: float
    boundary_norm: float
    energies: np.ndarray | None = None
    effective_hamiltonian: np.ndarray | None = None
    boundary_annihilation: np.ndarray | None = None


@dataclass
class SpinBosonWilsonNARGResult:
    """NARG eigenpairs and per-step diagnostics."""

    energies: np.ndarray
    vectors: np.ndarray
    steps: list[SpinBosonWilsonNARGStep]
    effective_hamiltonian: np.ndarray
    boundary_annihilation: np.ndarray | None
    sigma_z: np.ndarray | None = None
    magnetizations: np.ndarray | None = None


@dataclass
class SpinBosonModePES:
    """Adiabatic PESs seen when adding one Wilson-chain boson mode."""

    site: int
    q: np.ndarray
    surfaces: np.ndarray
    onsite_frequency: float
    coupling_norm: float


@dataclass
class SpinBosonFPESObservable:
    """Ground fixed-PES observables for one added Wilson-chain mode."""

    site: int
    q_left: float
    q_right: float
    well_separation: float
    q0: float
    barrier_height: float
    curvature: float
    energy_scale: float
    onsite_frequency: float
    coupling_norm: float


@dataclass
class SpinBosonFPESObservableScan:
    """Ground fixed-PES observables across Wilson-chain length."""

    sites: np.ndarray
    observables: list[SpinBosonFPESObservable]
    well_separations: np.ndarray
    q0: np.ndarray
    barrier_heights: np.ndarray
    curvatures: np.ndarray
    energy_scales: np.ndarray
    onsite_frequencies: np.ndarray
    coupling_norms: np.ndarray


@dataclass
class SpinBosonFPESAlphaScan:
    """FPES basin scan across spin-boson coupling strength."""

    alphas: np.ndarray
    sites: np.ndarray
    q0: np.ndarray
    barrier_heights: np.ndarray
    energy_scales: np.ndarray
    q0_slopes: np.ndarray
    endpoint_q0: np.ndarray
    q0_threshold: float
    pseudo_critical_alpha: float | None


def star_to_wilson_chain(frequencies, couplings):
    """Lanczos transform from star bath modes to Wilson-chain parameters."""
    frequencies = np.asarray(frequencies, dtype=float)
    couplings = np.asarray(couplings, dtype=float)
    if frequencies.ndim != 1 or couplings.ndim != 1:
        raise ValueError("frequencies and couplings must be one-dimensional.")
    if len(frequencies) != len(couplings):
        raise ValueError("frequencies and couplings must have the same length.")
    if len(frequencies) == 0:
        return np.array([]), np.array([]), 0.0, np.zeros((0, 0))

    impurity_coupling = float(np.linalg.norm(couplings))
    if impurity_coupling <= 0.0:
        raise ValueError("at least one star coupling must be nonzero.")

    vectors = []
    onsite = np.zeros(len(frequencies), dtype=float)
    hopping = np.zeros(max(0, len(frequencies) - 1), dtype=float)
    v_prev = np.zeros_like(couplings)
    v = couplings / impurity_coupling

    for n in range(len(frequencies)):
        vectors.append(v.copy())
        w = frequencies * v
        onsite[n] = float(np.dot(v, w))
        if n > 0:
            w -= hopping[n - 1] * v_prev
        w -= onsite[n] * v
        for q in vectors:
            w -= np.dot(q, w) * q
        beta = float(np.linalg.norm(w))
        if n < len(frequencies) - 1:
            hopping[n] = beta
            if beta <= 1e-14:
                vectors.extend(np.eye(len(frequencies))[len(vectors) :])
                break
            v_prev, v = v, w / beta

    transform = np.asarray(vectors[: len(frequencies)])
    return onsite, hopping, impurity_coupling, transform


def log_discretized_spin_boson_wilson_chain(
    nmodes: int,
    *,
    alpha: float,
    Lambda: float = 2.0,
    s: float = 1.0,
    omegac: float = 1.0,
    epsilon: float = 0.0,
    delta: float = 0.0,
) -> SpinBosonWilsonChain:
    """Log-discretize an Ohmic-like spin-boson bath and return a Wilson chain.

    The spectral density convention follows the existing impurity SBM code:
    ``J(omega) = 2 pi alpha omegac^(1-s) omega^s`` on ``[0, omegac]``.
    """
    nmodes = int(nmodes)
    if nmodes < 1:
        raise ValueError("nmodes must be positive.")
    if Lambda <= 1.0:
        raise ValueError("Lambda must be larger than one.")
    if s <= -1.0:
        raise ValueError("s must be larger than -1.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    n = np.arange(nmodes)
    frequencies = (
        (s + 1.0)
        / (s + 2.0)
        * (1.0 - Lambda ** (-s - 2.0))
        / (1.0 - Lambda ** (-s - 1.0))
        * omegac
        * Lambda ** (-n)
    )
    coupling2 = (
        2.0
        * np.pi
        * alpha
        / (s + 1.0)
        * omegac**2
        * (1.0 - Lambda ** (-s - 1.0))
        * Lambda ** (-n * (s + 1.0))
    )
    couplings = np.sqrt(coupling2)
    onsite, hopping, impurity_coupling, transform = star_to_wilson_chain(frequencies, couplings)
    return SpinBosonWilsonChain(
        onsite=onsite,
        hopping=hopping,
        impurity_coupling=impurity_coupling,
        epsilon=epsilon,
        delta=delta,
        star_frequencies=frequencies,
        star_couplings=couplings,
        star_to_chain=transform,
    )


def _kron_all(operators, *, sparse=False):
    if not operators:
        raise ValueError("operators must be non-empty.")
    out = csr_matrix(operators[0]) if sparse else np.asarray(operators[0])
    for op in operators[1:]:
        out = kron(out, csr_matrix(op), format="csr") if sparse else np.kron(out, op)
    return out


def _resolve_displacements(chain: SpinBosonWilsonChain, displacements):
    if displacements is None or displacements is False:
        return np.zeros(chain.nmodes, dtype=float)
    if isinstance(displacements, str):
        key = displacements.lower()
        if key in {"auto", "estimate", "classical"}:
            return chain.estimate_displacements()
        if key in {"none", "zero", "false"}:
            return np.zeros(chain.nmodes, dtype=float)
        raise ValueError("displacements must be None, an array, or 'auto'.")
    values = np.asarray(displacements, dtype=float)
    if values.shape != (chain.nmodes,):
        raise ValueError("displacements must have shape (chain.nmodes,).")
    return values


def _site_boson_operators(
    chain: SpinBosonWilsonChain,
    nboson: int,
    *,
    basis: str,
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
):
    shifts = _resolve_displacements(chain, displacements)
    return [
        local_boson_operators(
            nboson,
            basis=basis,
            displacement=shifts[site],
            parent_dim=parent_dim,
            symmetric_displacement=symmetric_displacement,
            dvr_qmax=dvr_qmax,
        )
        for site in range(chain.nmodes)
    ]


def spin_boson_wilson_hamiltonian(
    chain: SpinBosonWilsonChain,
    nboson: int,
    *,
    sparse=False,
    basis: str = "dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
):
    """Build the finite spin-boson Wilson-chain Hamiltonian."""
    nboson = int(nboson)
    if nboson < 1:
        raise ValueError("nboson must be positive.")
    identity_spin, _, _, z = spin_operators()
    site_ops = _site_boson_operators(
        chain,
        nboson,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    )
    dims = [2] + [nboson] * chain.nmodes
    dim = int(np.prod(dims))
    hamiltonian = csr_matrix((dim, dim), dtype=complex) if sparse else np.zeros((dim, dim), dtype=complex)

    identities = [ops[0] for ops in site_ops]
    ops = [chain.impurity_hamiltonian()] + identities
    hamiltonian = hamiltonian + _kron_all(ops, sparse=sparse)

    for mode, omega in enumerate(chain.onsite):
        ops = [identity_spin] + list(identities)
        ops[mode + 1] = site_ops[mode][3]
        hamiltonian = hamiltonian + omega * _kron_all(ops, sparse=sparse)

    if chain.nmodes:
        ops = [z] + list(identities)
        ops[1] = site_ops[0][1] + site_ops[0][2]
        hamiltonian = hamiltonian + 0.5 * chain.impurity_coupling * _kron_all(ops, sparse=sparse)

    for mode, hopping in enumerate(chain.hopping):
        ops = [identity_spin] + list(identities)
        ops[mode + 1] = site_ops[mode][2]
        ops[mode + 2] = site_ops[mode + 1][1]
        hamiltonian = hamiltonian + hopping * _kron_all(ops, sparse=sparse)
        ops = [identity_spin] + list(identities)
        ops[mode + 1] = site_ops[mode][1]
        ops[mode + 2] = site_ops[mode + 1][2]
        hamiltonian = hamiltonian + hopping * _kron_all(ops, sparse=sparse)

    return hamiltonian


def spin_boson_wilson_exact(
    chain: SpinBosonWilsonChain,
    nboson: int,
    *,
    nroots: int = 1,
    basis: str = "dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
):
    """Return lowest exact eigenpairs for a small finite Wilson chain."""
    hamiltonian = spin_boson_wilson_hamiltonian(
        chain,
        nboson,
        sparse=True,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    )
    nroots = min(int(nroots), hamiltonian.shape[0])
    if hamiltonian.shape[0] <= max(128, nroots + 2):
        values, vectors = np.linalg.eigh(hamiltonian.toarray())
        return values[:nroots], vectors[:, :nroots]
    values, vectors = eigsh(hamiltonian, k=nroots, which="SA")
    order = np.argsort(values)
    return values[order], vectors[:, order]


def spin_boson_wilson_exact_magnetization(
    chain: SpinBosonWilsonChain,
    nboson: int,
    *,
    nroots: int = 1,
    basis: str = "dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
):
    """Return exact eigenvalues and ``<sigma_z>`` for a small Wilson chain."""
    values, vectors = spin_boson_wilson_exact(
        chain,
        nboson,
        nroots=nroots,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    )
    _, _, _, z = spin_operators()
    site_ops = _site_boson_operators(
        chain,
        nboson,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    )
    operator = _kron_all([z] + [ops[0] for ops in site_ops], sparse=False)
    magnetizations = np.einsum("ik,ij,jk->k", vectors.conj(), operator, vectors)
    return values, np.real_if_close(magnetizations)


def _narg_block_before_site(
    chain: SpinBosonWilsonChain,
    site: int,
    *,
    nboson: int,
    bond_dim: int,
    basis: str = "dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
):
    site = int(site)
    if site < 0 or site >= chain.nmodes:
        raise ValueError("site must be between 0 and chain.nmodes - 1.")
    if site == 0:
        _, _, _, z = spin_operators()
        return chain.impurity_hamiltonian(), z, 2

    site_ops = _site_boson_operators(
        chain,
        nboson,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    )
    identity_b, b, bdag, number = site_ops[0]
    _, _, _, z = spin_operators()
    hamiltonian = (
        np.kron(chain.impurity_hamiltonian(), identity_b)
        + np.kron(np.eye(2, dtype=complex), chain.onsite[0] * number)
        + 0.5 * chain.impurity_coupling * np.kron(z, b + bdag)
    )
    keep = min(int(bond_dim), hamiltonian.shape[0])
    values, vectors = SpinBosonWilsonNARG._diagonalize(hamiltonian, keep)
    boundary_b = vectors.conj().T @ np.kron(np.eye(2, dtype=complex), b) @ vectors
    effective_hamiltonian = np.diag(values).astype(complex)

    for current_site in range(1, site):
        block_dim = effective_hamiltonian.shape[0]
        identity_b, b, bdag, number = site_ops[current_site]
        hamiltonian = (
            np.kron(effective_hamiltonian, identity_b)
            + np.kron(np.eye(block_dim, dtype=complex), chain.onsite[current_site] * number)
            + chain.hopping[current_site - 1]
            * (np.kron(boundary_b.conj().T, b) + np.kron(boundary_b, bdag))
        )
        keep = min(int(bond_dim), hamiltonian.shape[0])
        values, vectors = SpinBosonWilsonNARG._diagonalize(hamiltonian, keep)
        boundary_b = vectors.conj().T @ np.kron(np.eye(block_dim, dtype=complex), b) @ vectors
        effective_hamiltonian = np.diag(values).astype(complex)

    return effective_hamiltonian, boundary_b, effective_hamiltonian.shape[0]


def spin_boson_mode_pes(
    chain: SpinBosonWilsonChain,
    site: int,
    q,
    *,
    nboson: int,
    bond_dim: int,
    nlevels: int = 4,
    basis: str = "dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
    relative: bool = True,
    narg_result: SpinBosonWilsonNARGResult | None = None,
) -> SpinBosonModePES:
    """Adiabatic PESs obtained when adding Wilson-chain mode ``site``.

    The new oscillator coordinate is frozen at dimensionless ``q`` and its
    kinetic energy is omitted:

    ``H_PES(q) = H_block + 0.5 * omega_site * q**2 + q * V_block``.

    For ``site=0``, ``V_block = t0 sigma_z / sqrt(2)``.  For later sites,
    ``V_block = t_site (b_boundary + b_boundary^dagger) / sqrt(2)``.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or len(q) == 0:
        raise ValueError("q must be a non-empty one-dimensional grid.")
    if narg_result is not None and int(site) > 0:
        previous = narg_result.steps[int(site) - 1]
        if previous.effective_hamiltonian is None or previous.boundary_annihilation is None:
            raise ValueError("narg_result does not contain block snapshots.")
        h_block = previous.effective_hamiltonian
        boundary = previous.boundary_annihilation
        block_dim = h_block.shape[0]
    else:
        h_block, boundary, block_dim = _narg_block_before_site(
            chain,
            site,
            nboson=nboson,
            bond_dim=bond_dim,
            basis=basis,
            displacements=displacements,
            parent_dim=parent_dim,
            symmetric_displacement=symmetric_displacement,
            dvr_qmax=dvr_qmax,
        )
    if int(site) == 0:
        coupling = chain.impurity_coupling * boundary / np.sqrt(2.0)
    else:
        boundary_force = 0.5 * (boundary + boundary.T.conj())
        coupling = np.sqrt(2.0) * chain.hopping[int(site) - 1] * boundary_force

    nlevels = min(int(nlevels), int(block_dim))
    surfaces = np.empty((len(q), nlevels), dtype=float)
    identity = np.eye(int(block_dim), dtype=complex)
    for index, coordinate in enumerate(q):
        hq = (
            h_block
            + coordinate * coupling
            + 0.5 * chain.onsite[int(site)] * coordinate**2 * identity
        )
        values = np.linalg.eigvalsh(0.5 * (hq + hq.T.conj()))
        surfaces[index] = values[:nlevels]
    if relative:
        surfaces = surfaces - float(np.min(surfaces[:, 0]))
    return SpinBosonModePES(
        site=int(site),
        q=q,
        surfaces=surfaces,
        onsite_frequency=float(chain.onsite[int(site)]),
        coupling_norm=float(np.linalg.norm(coupling)),
    )


def _quadratic_curvature(q, values, index, *, window: int = 2):
    start = max(0, int(index) - int(window))
    stop = min(len(q), int(index) + int(window) + 1)
    if stop - start < 3:
        return np.nan
    coeffs = np.polyfit(q[start:stop], values[start:stop], 2)
    return float(2.0 * coeffs[0])


def _quadratic_minimum(q, values, index, *, window: int = 2):
    start = max(0, int(index) - int(window))
    stop = min(len(q), int(index) + int(window) + 1)
    if stop - start < 3:
        return float(q[index]), float(values[index]), np.nan
    coeffs = np.polyfit(q[start:stop], values[start:stop], 2)
    curvature = 2.0 * coeffs[0]
    if abs(coeffs[0]) <= 1e-14:
        return float(q[index]), float(values[index]), float(curvature)
    vertex = -coeffs[1] / (2.0 * coeffs[0])
    if vertex < q[start] or vertex > q[stop - 1]:
        vertex = float(q[index])
    energy = float(np.polyval(coeffs, vertex))
    return float(vertex), energy, float(curvature)


def extract_spin_boson_fpes_observable(
    pes: SpinBosonModePES,
    *,
    curvature_window: int = 2,
) -> SpinBosonFPESObservable:
    """Extract geometric observables from a ground-state adiabatic PES."""
    q = np.asarray(pes.q, dtype=float)
    ground = np.asarray(pes.surfaces[:, 0], dtype=float)
    ground = ground - float(np.min(ground))

    left_indices = np.flatnonzero(q < 0.0)
    right_indices = np.flatnonzero(q > 0.0)
    if len(left_indices) == 0 or len(right_indices) == 0:
        raise ValueError("q grid must contain both negative and positive points.")

    left_index = int(left_indices[np.argmin(ground[left_indices])])
    right_index = int(right_indices[np.argmin(ground[right_indices])])
    q_left, left_energy, left_curvature = _quadratic_minimum(
        q,
        ground,
        left_index,
        window=curvature_window,
    )
    q_right, right_energy, right_curvature = _quadratic_minimum(
        q,
        ground,
        right_index,
        window=curvature_window,
    )
    min_energy = 0.5 * (left_energy + right_energy)
    barrier = float(np.interp(0.0, q, ground) - min_energy)
    energy_scale = float(np.max(ground) - np.min(ground))

    curvature = float(np.nanmean([left_curvature, right_curvature]))

    return SpinBosonFPESObservable(
        site=int(pes.site),
        q_left=q_left,
        q_right=q_right,
        well_separation=float(q_right - q_left),
        q0=0.5 * float(abs(q_left) + abs(q_right)),
        barrier_height=max(barrier, 0.0),
        curvature=curvature,
        energy_scale=energy_scale,
        onsite_frequency=float(pes.onsite_frequency),
        coupling_norm=float(pes.coupling_norm),
    )


def scan_spin_boson_fpes_observables(
    chain: SpinBosonWilsonChain,
    sites,
    q,
    *,
    nboson: int,
    bond_dim: int,
    basis: str = "dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
    curvature_window: int = 2,
) -> SpinBosonFPESObservableScan:
    """Extract ground-FPES observables for several added Wilson modes."""
    sites = np.asarray(sites, dtype=int)
    narg_result = SpinBosonWilsonNARG(
        chain,
        nboson=nboson,
        bond_dim=bond_dim,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    ).run(nroots=1)
    observables = []
    for site in sites:
        pes = spin_boson_mode_pes(
            chain,
            int(site),
            q,
            nboson=nboson,
            bond_dim=bond_dim,
            nlevels=1,
            basis=basis,
            displacements=displacements,
            parent_dim=parent_dim,
            symmetric_displacement=symmetric_displacement,
            dvr_qmax=dvr_qmax,
            narg_result=narg_result,
        )
        observables.append(
            extract_spin_boson_fpes_observable(
                pes,
                curvature_window=curvature_window,
            )
        )

    return SpinBosonFPESObservableScan(
        sites=sites,
        observables=observables,
        well_separations=np.asarray([obs.well_separation for obs in observables]),
        q0=np.asarray([obs.q0 for obs in observables]),
        barrier_heights=np.asarray([obs.barrier_height for obs in observables]),
        curvatures=np.asarray([obs.curvature for obs in observables]),
        energy_scales=np.asarray([obs.energy_scale for obs in observables]),
        onsite_frequencies=np.asarray([obs.onsite_frequency for obs in observables]),
        coupling_norms=np.asarray([obs.coupling_norm for obs in observables]),
    )


def _wilson_log_slope(sites, values, *, Lambda: float = 2.0):
    sites = np.asarray(sites, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (values > 0.0) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return np.nan
    x = sites[mask] * np.log(float(Lambda))
    y = np.log(values[mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def scan_spin_boson_fpes_alpha(
    alphas,
    sites,
    q,
    *,
    nmodes: int,
    nboson: int,
    bond_dim: int,
    Lambda: float = 2.0,
    s: float = 0.5,
    omegac: float = 1.0,
    delta: float = 0.1,
    q0_threshold: float = 0.5,
    basis: str = "sine-dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
) -> SpinBosonFPESAlphaScan:
    """Scan alpha and locate the FPES single-well/double-well crossover."""
    alphas = np.asarray(alphas, dtype=float)
    sites = np.asarray(sites, dtype=int)
    if alphas.ndim != 1 or len(alphas) == 0:
        raise ValueError("alphas must be a non-empty one-dimensional sequence.")
    if sites.ndim != 1 or len(sites) == 0:
        raise ValueError("sites must be a non-empty one-dimensional sequence.")
    if np.max(sites) >= int(nmodes):
        raise ValueError("all sites must be smaller than nmodes.")

    q0 = np.empty((len(alphas), len(sites)), dtype=float)
    barriers = np.empty_like(q0)
    energy_scales = np.empty_like(q0)
    q0_slopes = np.empty(len(alphas), dtype=float)
    endpoint_q0 = np.empty(len(alphas), dtype=float)

    for index, alpha in enumerate(alphas):
        chain = log_discretized_spin_boson_wilson_chain(
            nmodes,
            alpha=float(alpha),
            Lambda=Lambda,
            s=s,
            omegac=omegac,
            epsilon=0.0,
            delta=delta,
        )
        scan = scan_spin_boson_fpes_observables(
            chain,
            sites,
            q,
            nboson=nboson,
            bond_dim=bond_dim,
            basis=basis,
            displacements=displacements,
            parent_dim=parent_dim,
            symmetric_displacement=symmetric_displacement,
            dvr_qmax=dvr_qmax,
        )
        q0[index] = scan.q0
        barriers[index] = scan.barrier_heights
        energy_scales[index] = scan.energy_scales
        q0_slopes[index] = _wilson_log_slope(sites, scan.q0, Lambda=Lambda)
        endpoint_q0[index] = float(scan.q0[-1])

    alpha_c = None
    above = endpoint_q0 >= float(q0_threshold)
    crossing = np.flatnonzero(above[1:] & ~above[:-1])
    if len(crossing):
        left = int(crossing[0])
        right = left + 1
        denom = endpoint_q0[right] - endpoint_q0[left]
        if abs(denom) <= 1e-14:
            alpha_c = float(0.5 * (alphas[left] + alphas[right]))
        else:
            fraction = (float(q0_threshold) - endpoint_q0[left]) / denom
            alpha_c = float(alphas[left] + fraction * (alphas[right] - alphas[left]))
    elif np.any(above):
        alpha_c = float(alphas[int(np.flatnonzero(above)[0])])

    return SpinBosonFPESAlphaScan(
        alphas=alphas,
        sites=sites,
        q0=q0,
        barrier_heights=barriers,
        energy_scales=energy_scales,
        q0_slopes=q0_slopes,
        endpoint_q0=endpoint_q0,
        q0_threshold=float(q0_threshold),
        pseudo_critical_alpha=alpha_c,
    )


@dataclass
class SpinBosonCriticalScan:
    """Coupling scan data for finite Wilson-chain critical diagnostics."""

    alphas: np.ndarray
    energies: np.ndarray
    gaps: np.ndarray
    magnetizations: np.ndarray
    susceptibilities: np.ndarray | None = None
    pseudo_critical_alpha: float | None = None


@dataclass
class SpinBosonFiniteSizeGapScan:
    """Finite-size gap-collapse diagnostics across Wilson-chain lengths."""

    nmodes: np.ndarray
    alphas: np.ndarray
    gaps: np.ndarray
    threshold: float
    threshold_alphas: np.ndarray
    minimum_gap_alphas: np.ndarray
    minimum_gaps: np.ndarray


@dataclass
class SpinBosonFixedPointScan:
    """Stationarity scan for rescaled Wilson-chain NARG spectra."""

    alphas: np.ndarray
    scores: np.ndarray
    best_alpha: float
    spectra: np.ndarray
    rescaled_gaps: np.ndarray
    nlevels: int
    late_steps: int
    Lambda: float
    rescale_power: float


@dataclass
class SpinBosonFixedPointFlowScan:
    """Per-alpha NARG fixed-point endpoint flow diagnostics."""

    alphas: np.ndarray
    spectra: np.ndarray
    endpoint_spectra: np.ndarray
    drift_scores: np.ndarray
    endpoint_changes: np.ndarray
    crossover_alpha: float | None
    nlevels: int
    late_steps: int
    Lambda: float
    rescale_power: float


@dataclass
class PowerLawFit:
    """Log-log power-law fit ``y = amplitude * x**exponent``."""

    exponent: float
    amplitude: float
    intercept: float
    r2: float
    x: np.ndarray
    y: np.ndarray


def _fit_power_law(x, y) -> PowerLawFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two positive finite points are required for a power-law fit.")
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return PowerLawFit(
        exponent=float(slope),
        amplitude=float(np.exp(intercept)),
        intercept=float(intercept),
        r2=r2,
        x=x[mask],
        y=y[mask],
    )


def scan_spin_boson_alpha(
    alphas,
    *,
    nmodes: int,
    nboson: int,
    bond_dim: int,
    Lambda: float = 2.0,
    s: float = 0.5,
    omegac: float = 1.0,
    epsilon: float = 1e-4,
    delta: float = 0.1,
    nroots: int = 2,
    basis: str = "dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
) -> SpinBosonCriticalScan:
    """Scan spin-boson coupling and return finite-chain critical diagnostics.

    A small bias ``epsilon`` turns the order parameter ``<sigma_z>`` into a
    stable finite-size diagnostic.  The pseudocritical coupling is estimated
    from the largest finite-difference susceptibility ``d<sigma_z>/d epsilon``
    when a nonzero bias is used.
    """
    alphas = np.asarray(alphas, dtype=float)
    energies = np.empty((len(alphas), int(nroots)), dtype=float)
    gaps = np.empty(len(alphas), dtype=float)
    magnetizations = np.empty(len(alphas), dtype=float)
    susceptibilities = np.empty(len(alphas), dtype=float) if abs(epsilon) > 0.0 else None

    for index, alpha in enumerate(alphas):
        chain = log_discretized_spin_boson_wilson_chain(
            nmodes,
            alpha=float(alpha),
            Lambda=Lambda,
            s=s,
            omegac=omegac,
            epsilon=epsilon,
            delta=delta,
        )
        result = SpinBosonWilsonNARG(
            chain,
            nboson=nboson,
            bond_dim=bond_dim,
            basis=basis,
            displacements=displacements,
            parent_dim=parent_dim,
            symmetric_displacement=symmetric_displacement,
            dvr_qmax=dvr_qmax,
        ).run(nroots=nroots)
        energies[index, : len(result.energies)] = result.energies
        gaps[index] = result.energies[1] - result.energies[0] if len(result.energies) > 1 else np.nan
        magnetizations[index] = float(np.real(result.magnetizations[0]))

        if susceptibilities is not None:
            chain_minus = log_discretized_spin_boson_wilson_chain(
                nmodes,
                alpha=float(alpha),
                Lambda=Lambda,
                s=s,
                omegac=omegac,
                epsilon=-epsilon,
                delta=delta,
            )
            minus = SpinBosonWilsonNARG(
                chain_minus,
                nboson=nboson,
                bond_dim=bond_dim,
                basis=basis,
                displacements=displacements,
                parent_dim=parent_dim,
                symmetric_displacement=symmetric_displacement,
                dvr_qmax=dvr_qmax,
            ).run(nroots=1)
            susceptibilities[index] = (result.magnetizations[0] - minus.magnetizations[0]) / (2.0 * epsilon)

    pseudo = None
    if susceptibilities is not None and len(susceptibilities):
        pseudo = float(alphas[int(np.nanargmax(np.abs(susceptibilities)))])
    return SpinBosonCriticalScan(
        alphas=alphas,
        energies=energies,
        gaps=gaps,
        magnetizations=magnetizations,
        susceptibilities=susceptibilities,
        pseudo_critical_alpha=pseudo,
    )


def scan_spin_boson_gap_thresholds(
    nmodes_list,
    alphas,
    *,
    nboson: int,
    bond_dim,
    gap_threshold: float = 1e-9,
    Lambda: float = 2.0,
    s: float = 0.5,
    omegac: float = 1.0,
    delta: float = 0.1,
    basis: str = "sine-dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
) -> SpinBosonFiniteSizeGapScan:
    """Scan unbiased finite-chain gaps and locate pseudocritical thresholds.

    For the unbiased spin-boson chain, the lowest parity doublet gap is often
    the cleanest finite-size diagnostic.  ``threshold_alphas`` records the
    first coupling where the gap drops below ``gap_threshold``; if no scanned
    coupling crosses that threshold, the value is ``nan``.  ``minimum_gap_*``
    is reported separately because the smallest gap can sit deep inside the
    localized plateau rather than near the onset.
    """
    nmodes = np.asarray(nmodes_list, dtype=int)
    alphas = np.asarray(alphas, dtype=float)
    if nmodes.ndim != 1 or len(nmodes) == 0:
        raise ValueError("nmodes_list must be a non-empty one-dimensional sequence.")
    if alphas.ndim != 1 or len(alphas) == 0:
        raise ValueError("alphas must be a non-empty one-dimensional sequence.")
    if gap_threshold <= 0.0:
        raise ValueError("gap_threshold must be positive.")

    if np.isscalar(bond_dim):
        bond_dims = {int(nmode): int(bond_dim) for nmode in nmodes}
    else:
        values = np.asarray(bond_dim, dtype=int)
        if values.shape != nmodes.shape:
            raise ValueError("bond_dim must be a scalar or have the same shape as nmodes_list.")
        bond_dims = {int(nmode): int(dim) for nmode, dim in zip(nmodes, values)}

    gaps = np.empty((len(nmodes), len(alphas)), dtype=float)
    threshold_alphas = np.full(len(nmodes), np.nan, dtype=float)
    minimum_gap_alphas = np.empty(len(nmodes), dtype=float)
    minimum_gaps = np.empty(len(nmodes), dtype=float)

    for row, nmode in enumerate(nmodes):
        scan = scan_spin_boson_alpha(
            alphas,
            nmodes=int(nmode),
            nboson=nboson,
            bond_dim=bond_dims[int(nmode)],
            Lambda=Lambda,
            s=s,
            omegac=omegac,
            epsilon=0.0,
            delta=delta,
            nroots=2,
            basis=basis,
            displacements=displacements,
            parent_dim=parent_dim,
            symmetric_displacement=symmetric_displacement,
            dvr_qmax=dvr_qmax,
        )
        gaps[row] = scan.gaps
        crossed = np.flatnonzero(scan.gaps < gap_threshold)
        if len(crossed):
            threshold_alphas[row] = float(alphas[int(crossed[0])])
        minimum = int(np.nanargmin(scan.gaps))
        minimum_gap_alphas[row] = float(alphas[minimum])
        minimum_gaps[row] = float(scan.gaps[minimum])

    return SpinBosonFiniteSizeGapScan(
        nmodes=nmodes,
        alphas=alphas,
        gaps=gaps,
        threshold=float(gap_threshold),
        threshold_alphas=threshold_alphas,
        minimum_gap_alphas=minimum_gap_alphas,
        minimum_gaps=minimum_gaps,
    )


def narg_rescaled_spectrum_flow(
    result: SpinBosonWilsonNARGResult,
    *,
    Lambda: float = 2.0,
    nlevels: int = 4,
    rescale_power: float = 1.0,
) -> np.ndarray:
    """Return step-rescaled low-energy gaps from a NARG run.

    The returned array has shape ``(nsteps, nlevels - 1)`` and contains
    ``(E_i - E_0) * Lambda**(rescale_power * site)`` for each stored step.
    """
    nlevels = int(nlevels)
    if nlevels < 2:
        raise ValueError("nlevels must be at least 2.")
    rows = []
    width = nlevels - 1
    for step in result.steps:
        if step.energies is None or len(step.energies) < 2:
            row = np.full(width, np.nan, dtype=float)
        else:
            values = np.asarray(step.energies[:nlevels], dtype=float)
            gaps = values[1:] - values[0]
            row = np.full(width, np.nan, dtype=float)
            row[: len(gaps)] = gaps * float(Lambda) ** (float(rescale_power) * step.site)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _fixed_point_score(flow, *, late_steps: int = 3):
    flow = np.asarray(flow, dtype=float)
    late_steps = int(late_steps)
    if flow.ndim != 2:
        raise ValueError("flow must be two-dimensional.")
    if late_steps < 2:
        raise ValueError("late_steps must be at least 2.")
    if len(flow) < late_steps:
        return np.inf
    tail = flow[-late_steps:]
    if not np.all(np.isfinite(tail)):
        return np.inf
    scale = np.maximum(np.mean(np.abs(tail), axis=0), 1e-14)
    relative = np.diff(tail, axis=0) / scale
    return float(np.sqrt(np.mean(relative**2)))


def scan_spin_boson_fixed_point(
    alphas,
    *,
    nmodes: int,
    nboson: int,
    bond_dim: int,
    Lambda: float = 2.0,
    s: float = 0.5,
    omegac: float = 1.0,
    delta: float = 0.1,
    nlevels: int = 4,
    late_steps: int = 3,
    rescale_power: float = 1.0,
    basis: str = "sine-dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
) -> SpinBosonFixedPointScan:
    """Find the alpha whose rescaled NARG spectrum is most stationary."""
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or len(alphas) == 0:
        raise ValueError("alphas must be a non-empty one-dimensional sequence.")
    flows = []
    scores = np.empty(len(alphas), dtype=float)
    final_gaps = np.empty((len(alphas), int(nlevels) - 1), dtype=float)
    for index, alpha in enumerate(alphas):
        chain = log_discretized_spin_boson_wilson_chain(
            nmodes,
            alpha=float(alpha),
            Lambda=Lambda,
            s=s,
            omegac=omegac,
            epsilon=0.0,
            delta=delta,
        )
        result = SpinBosonWilsonNARG(
            chain,
            nboson=nboson,
            bond_dim=bond_dim,
            basis=basis,
            displacements=displacements,
            parent_dim=parent_dim,
            symmetric_displacement=symmetric_displacement,
            dvr_qmax=dvr_qmax,
        ).run(nroots=nlevels)
        flow = narg_rescaled_spectrum_flow(
            result,
            Lambda=Lambda,
            nlevels=nlevels,
            rescale_power=rescale_power,
        )
        flows.append(flow)
        scores[index] = _fixed_point_score(flow, late_steps=late_steps)
        final_gaps[index] = flow[-1]

    best = int(np.nanargmin(scores))
    return SpinBosonFixedPointScan(
        alphas=alphas,
        scores=scores,
        best_alpha=float(alphas[best]),
        spectra=np.asarray(flows, dtype=float),
        rescaled_gaps=final_gaps,
        nlevels=int(nlevels),
        late_steps=int(late_steps),
        Lambda=float(Lambda),
        rescale_power=float(rescale_power),
    )


def scan_spin_boson_fixed_point_flows(
    alphas,
    *,
    nmodes: int,
    nboson: int,
    bond_dim: int,
    Lambda: float = 2.0,
    s: float = 0.5,
    omegac: float = 1.0,
    delta: float = 0.1,
    nlevels: int = 4,
    late_steps: int = 3,
    rescale_power: float = 1.0,
    basis: str = "sine-dvr",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
) -> SpinBosonFixedPointFlowScan:
    """Track the NARG endpoint reached by each alpha.

    This is the RG diagnostic to use before extracting critical data: each
    coupling has its own flow along Wilson-chain length.  The critical coupling
    is inferred from a change in endpoint basin, not from a single alpha having
    the smallest drift score.
    """
    scan = scan_spin_boson_fixed_point(
        alphas,
        nmodes=nmodes,
        nboson=nboson,
        bond_dim=bond_dim,
        Lambda=Lambda,
        s=s,
        omegac=omegac,
        delta=delta,
        nlevels=nlevels,
        late_steps=late_steps,
        rescale_power=rescale_power,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    )

    endpoints = np.mean(scan.spectra[:, -int(late_steps) :, :], axis=1)
    endpoint_changes = np.empty(max(0, len(scan.alphas) - 1), dtype=float)
    for index in range(len(endpoint_changes)):
        left = endpoints[index]
        right = endpoints[index + 1]
        scale = max(0.5 * (np.linalg.norm(left) + np.linalg.norm(right)), 1e-14)
        endpoint_changes[index] = float(np.linalg.norm(right - left) / scale)

    crossover_alpha = None
    if len(endpoint_changes) and np.any(np.isfinite(endpoint_changes)):
        index = int(np.nanargmax(endpoint_changes))
        crossover_alpha = float(0.5 * (scan.alphas[index] + scan.alphas[index + 1]))

    return SpinBosonFixedPointFlowScan(
        alphas=scan.alphas,
        spectra=scan.spectra,
        endpoint_spectra=endpoints,
        drift_scores=scan.scores,
        endpoint_changes=endpoint_changes,
        crossover_alpha=crossover_alpha,
        nlevels=scan.nlevels,
        late_steps=scan.late_steps,
        Lambda=scan.Lambda,
        rescale_power=scan.rescale_power,
    )


def fit_order_parameter_exponent(alphas, magnetizations, alpha_c):
    """Fit ``m ~ (alpha - alpha_c)**beta`` on the localized side."""
    alphas = np.asarray(alphas, dtype=float)
    magnetizations = np.asarray(magnetizations, dtype=float)
    return _fit_power_law(alphas - float(alpha_c), np.abs(magnetizations))


def fit_gap_exponent(alphas, gaps, alpha_c):
    """Fit finite-chain gap trend ``gap ~ |alpha - alpha_c|**(nu*z)``."""
    alphas = np.asarray(alphas, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    return _fit_power_law(np.abs(alphas - float(alpha_c)), gaps)


def fit_field_exponent(epsilons, magnetizations):
    """Fit critical isotherm ``m ~ epsilon**(1/delta)``."""
    return _fit_power_law(np.abs(epsilons), np.abs(magnetizations))


class SpinBosonWilsonNARG:
    """Sequential NARG for a finite spin-boson Wilson chain."""

    def __init__(
        self,
        chain: SpinBosonWilsonChain,
        *,
        nboson: int = 8,
        bond_dim: int = 32,
        basis: str = "dvr",
        displacements=None,
        parent_dim: int | None = None,
        symmetric_displacement: bool = True,
        dvr_qmax: float = 8.0,
    ):
        self.chain = chain
        self.nboson = int(nboson)
        self.bond_dim = int(bond_dim)
        self.basis = str(basis).lower()
        self.displacements = displacements
        self.parent_dim = parent_dim
        self.symmetric_displacement = bool(symmetric_displacement)
        self.dvr_qmax = float(dvr_qmax)
        if self.nboson < 1:
            raise ValueError("nboson must be positive.")
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")
        _site_boson_operators(
            self.chain,
            self.nboson,
            basis=self.basis,
            displacements=self.displacements,
            parent_dim=self.parent_dim,
            symmetric_displacement=self.symmetric_displacement,
            dvr_qmax=self.dvr_qmax,
        )

    @staticmethod
    def _diagonalize(hamiltonian, keep):
        keep = min(int(keep), hamiltonian.shape[0])
        values, vectors = np.linalg.eigh(0.5 * (hamiltonian + hamiltonian.T.conj()))
        return values[:keep], vectors[:, :keep]

    def run(self, nroots: int = 1) -> SpinBosonWilsonNARGResult:
        chain = self.chain
        nroots = int(nroots)
        if chain.nmodes == 0:
            values, vectors = self._diagonalize(chain.impurity_hamiltonian(), nroots)
            _, _, _, z = spin_operators()
            magnetizations = np.einsum("ik,ij,jk->k", vectors.conj(), z, vectors)
            return SpinBosonWilsonNARGResult(
                values,
                vectors,
                [],
                np.diag(values),
                None,
                sigma_z=z,
                magnetizations=np.real_if_close(magnetizations),
            )

        site_ops = _site_boson_operators(
            chain,
            self.nboson,
            basis=self.basis,
            displacements=self.displacements,
            parent_dim=self.parent_dim,
            symmetric_displacement=self.symmetric_displacement,
            dvr_qmax=self.dvr_qmax,
        )
        identity_b, b, bdag, number = site_ops[0]
        _, _, _, z = spin_operators()
        steps = []

        hamiltonian = (
            np.kron(chain.impurity_hamiltonian(), identity_b)
            + np.kron(np.eye(2, dtype=complex), chain.onsite[0] * number)
            + 0.5 * chain.impurity_coupling * np.kron(z, b + bdag)
        )
        keep = min(self.bond_dim, hamiltonian.shape[0])
        values, vectors = self._diagonalize(hamiltonian, keep)
        boundary_b = vectors.conj().T @ np.kron(np.eye(2, dtype=complex), b) @ vectors
        sigma_z = vectors.conj().T @ np.kron(z, identity_b) @ vectors
        effective_hamiltonian = np.diag(values).astype(complex)
        steps.append(
            SpinBosonWilsonNARGStep(
                site=0,
                product_dim=hamiltonian.shape[0],
                kept=len(values),
                lowest_energy=float(values[0]),
                boundary_norm=float(np.linalg.norm(boundary_b)),
                energies=values.copy(),
                effective_hamiltonian=effective_hamiltonian.copy(),
                boundary_annihilation=boundary_b.copy(),
            )
        )

        for site in range(1, chain.nmodes):
            block_dim = effective_hamiltonian.shape[0]
            identity_b, b, bdag, number = site_ops[site]
            hamiltonian = (
                np.kron(effective_hamiltonian, identity_b)
                + np.kron(np.eye(block_dim, dtype=complex), chain.onsite[site] * number)
                + chain.hopping[site - 1]
                * (np.kron(boundary_b.conj().T, b) + np.kron(boundary_b, bdag))
            )
            keep = min(self.bond_dim, hamiltonian.shape[0])
            values, vectors = self._diagonalize(hamiltonian, keep)
            boundary_b = vectors.conj().T @ np.kron(np.eye(block_dim, dtype=complex), b) @ vectors
            sigma_z = vectors.conj().T @ np.kron(sigma_z, identity_b) @ vectors
            effective_hamiltonian = np.diag(values).astype(complex)
            steps.append(
                SpinBosonWilsonNARGStep(
                    site=site,
                    product_dim=hamiltonian.shape[0],
                    kept=len(values),
                    lowest_energy=float(values[0]),
                    boundary_norm=float(np.linalg.norm(boundary_b)),
                    energies=values.copy(),
                    effective_hamiltonian=effective_hamiltonian.copy(),
                    boundary_annihilation=boundary_b.copy(),
                )
            )

        magnetizations = np.diag(sigma_z)[:nroots]
        return SpinBosonWilsonNARGResult(
            energies=values[:nroots],
            vectors=vectors[:, :nroots],
            steps=steps,
            effective_hamiltonian=effective_hamiltonian,
            boundary_annihilation=boundary_b,
            sigma_z=sigma_z,
            magnetizations=np.real_if_close(magnetizations),
        )


__all__ = [
    "SpinBosonWilsonChain",
    "SpinBosonCriticalScan",
    "SpinBosonFiniteSizeGapScan",
    "SpinBosonFixedPointScan",
    "SpinBosonFixedPointFlowScan",
    "SpinBosonModePES",
    "SpinBosonFPESObservable",
    "SpinBosonFPESObservableScan",
    "SpinBosonFPESAlphaScan",
    "SpinBosonWilsonNARG",
    "SpinBosonWilsonNARGResult",
    "SpinBosonWilsonNARGStep",
    "PowerLawFit",
    "boson_displaced_dvr_operators",
    "boson_dvr_operators",
    "fit_field_exponent",
    "fit_gap_exponent",
    "fit_order_parameter_exponent",
    "extract_spin_boson_fpes_observable",
    "boson_operators",
    "local_boson_operators",
    "log_discretized_spin_boson_wilson_chain",
    "narg_rescaled_spectrum_flow",
    "scan_spin_boson_fixed_point",
    "scan_spin_boson_fixed_point_flows",
    "scan_spin_boson_fpes_alpha",
    "scan_spin_boson_fpes_observables",
    "scan_spin_boson_gap_thresholds",
    "scan_spin_boson_alpha",
    "sine_dvr_boson_operators",
    "spin_boson_mode_pes",
    "spin_boson_wilson_exact",
    "spin_boson_wilson_exact_magnetization",
    "spin_boson_wilson_hamiltonian",
    "spin_operators",
    "star_to_wilson_chain",
]
