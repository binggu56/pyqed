"""NARG for spin-boson Wilson chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh, lobpcg

from pyqed.models.impurity.spin_boson import (
    SpinBosonWilsonChain,
    log_discretized_spin_boson_wilson_chain,
)
from pyqed.models.impurity.wilson import star_to_wilson_chain


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
    sigma_z: np.ndarray | None = None
    product_vectors: np.ndarray | None = None
    rescale_factor: float = 1.0
    nrg_rescaled: bool = False


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
    nrg_rescaled: bool = False
    nrg_Lambda: float | None = None
    nrg_rescale_power: float = 1.0
    nrg_scale: str = "lambda"


@dataclass
class SpinBosonWilsonAdiabaticNARGStep(SpinBosonWilsonNARGStep):
    """One orthonormal conditional-basis spin-boson NARG step.

    The basis is ``|q_i> |A_a(q_i)>``.  It is orthonormal because the sine-DVR
    coordinate states are orthonormal, so the overlap matrix is the identity.
    Nonadiabatic effects enter through the kinetic and momentum-coupling matrix
    elements dressed by conditional-state overlaps.
    """

    conditional_dim: int = 0
    q_grid: np.ndarray | None = None
    conditional_energies: np.ndarray | None = None


@dataclass
class SpinBosonWilsonAdiabaticNARGResult(SpinBosonWilsonNARGResult):
    """Result of orthonormal conditional-basis spin-boson NARG."""


@dataclass
class SpinBosonNARGStepObservables:
    """Low-energy observables extracted from stored Wilson-chain NARG steps."""

    nvalues: np.ndarray
    energies: np.ndarray
    gaps: np.ndarray
    magnetizations: np.ndarray
    kept: np.ndarray
    product_dims: np.ndarray
    boundary_norms: np.ndarray
    rescale_factors: np.ndarray


@dataclass
class SpinBosonWilsonDMRGResult:
    """Finite-chain DMRG result for a spin-boson Wilson chain."""

    energies: np.ndarray
    magnetization: float
    dmrg: object
    mpo: list[np.ndarray]
    sigma_z_mpo: list[np.ndarray]
    dims: list[int]


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


def _nearest_neighbor_mpo(local_terms, bond_channels):
    """Build an OBC MPO for local terms plus nearest-neighbor product terms."""
    local_terms = [np.asarray(term, dtype=complex) for term in local_terms]
    if not local_terms:
        raise ValueError("local_terms must be non-empty.")
    if len(bond_channels) != len(local_terms) - 1:
        raise ValueError("bond_channels must have length len(local_terms) - 1.")

    max_channels = max([len(channels) for channels in bond_channels] + [0])
    bond_dim = max_channels + 2
    idle = bond_dim - 1
    factors = []
    for site, local in enumerate(local_terms):
        dim = local.shape[0]
        identity = np.eye(dim, dtype=complex)
        if site == 0:
            tensor = np.zeros((1, bond_dim, dim, dim), dtype=complex)
            tensor[0, 0] = local
            for channel, (left_op, _right_op) in enumerate(bond_channels[0]):
                tensor[0, 1 + channel] = left_op
            tensor[0, idle] = identity
        elif site == len(local_terms) - 1:
            tensor = np.zeros((bond_dim, 1, dim, dim), dtype=complex)
            tensor[0, 0] = identity
            for channel, (_left_op, right_op) in enumerate(bond_channels[site - 1]):
                tensor[1 + channel, 0] = right_op
            tensor[idle, 0] = local
        else:
            tensor = np.zeros((bond_dim, bond_dim, dim, dim), dtype=complex)
            tensor[0, 0] = identity
            for channel, (_left_op, right_op) in enumerate(bond_channels[site - 1]):
                tensor[1 + channel, 0] = right_op
            tensor[idle, 0] = local
            for channel, (left_op, _right_op) in enumerate(bond_channels[site]):
                tensor[idle, 1 + channel] = left_op
            tensor[idle, idle] = identity
        factors.append(tensor)
    return factors


def spin_boson_wilson_mpo(
    chain: SpinBosonWilsonChain,
    nboson: int,
    *,
    basis: str = "fock",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
):
    """Return the finite spin-boson Wilson-chain Hamiltonian as an MPO."""
    from pyqed.mps.mps import MPO

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
    local_terms = [chain.impurity_hamiltonian()]
    local_terms.extend(float(omega) * ops[3] for omega, ops in zip(chain.onsite, site_ops))

    bond_channels = []
    if chain.nmodes:
        x0 = site_ops[0][1] + site_ops[0][2]
        bond_channels.append([(0.5 * chain.impurity_coupling * z, x0)])
    for mode, hopping in enumerate(chain.hopping):
        left_ops = site_ops[mode]
        right_ops = site_ops[mode + 1]
        bond_channels.append(
            [
                (float(hopping) * left_ops[2], right_ops[1]),
                (float(hopping) * left_ops[1], right_ops[2]),
            ]
        )
    if not chain.nmodes:
        local_terms[0] = local_terms[0] + 0.0 * identity_spin
    return MPO(_nearest_neighbor_mpo(local_terms, bond_channels))


def spin_boson_sigma_z_mpo(chain: SpinBosonWilsonChain, nboson: int):
    """Return a product MPO for the impurity ``sigma_z`` operator."""
    from pyqed.mps.mps import MPO

    _, _, _, z = spin_operators()
    dims = [2] + [int(nboson)] * chain.nmodes
    factors = []
    for site, dim in enumerate(dims):
        operator = z if site == 0 else np.eye(dim, dtype=complex)
        factors.append(operator.reshape(1, 1, dim, dim))
    return MPO(factors)


def spin_boson_product_mps(
    chain: SpinBosonWilsonChain,
    nboson: int,
    *,
    basis: str = "fock",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
):
    """Return a separable spin/oscillator initial state for Wilson-chain DMRG."""
    from pyqed.mps.mps import MPS

    spin_values, spin_vectors = np.linalg.eigh(chain.impurity_hamiltonian())
    spin_state = spin_vectors[:, int(np.argmin(spin_values))]
    factors = [spin_state.reshape(1, 2, 1).astype(complex)]

    site_ops = _site_boson_operators(
        chain,
        nboson,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    )
    for _identity, _b, _bdag, oscillator in site_ops:
        values, vectors = np.linalg.eigh(0.5 * (oscillator + oscillator.T.conj()))
        local_state = vectors[:, int(np.argmin(values))]
        factors.append(local_state.reshape(1, int(nboson), 1).astype(complex))
    return MPS(factors, labels=["lv", "p", "rv"]).normalize()


def spin_boson_wilson_dmrg(
    chain: SpinBosonWilsonChain,
    nboson: int,
    *,
    bond_dim: int = 32,
    nsweeps: int = 8,
    nstates: int = 1,
    basis: str = "fock",
    displacements=None,
    parent_dim: int | None = None,
    symmetric_displacement: bool = True,
    dvr_qmax: float = 8.0,
    init_guess=None,
    sweep_tol: float = 1e-7,
    davidson_tol: float = 1e-7,
    davidson_max_iter: int = 60,
    noise: float = 1e-5,
    noise_decay: float = 0.25,
    verbose: int = 0,
    not_conv_err: bool = False,
):
    """Run finite two-site DMRG on a spin-boson Wilson-chain MPO."""
    from pyqed.mps.dmrg import DMRG

    mpo = spin_boson_wilson_mpo(
        chain,
        nboson,
        basis=basis,
        displacements=displacements,
        parent_dim=parent_dim,
        symmetric_displacement=symmetric_displacement,
        dvr_qmax=dvr_qmax,
    )
    if init_guess is None:
        init_guess = spin_boson_product_mps(
            chain,
            nboson,
            basis=basis,
            displacements=displacements,
            parent_dim=parent_dim,
            symmetric_displacement=symmetric_displacement,
            dvr_qmax=dvr_qmax,
        )
    solver = DMRG(
        mpo,
        D=int(bond_dim),
        init_guess=init_guess,
        nsweeps=int(nsweeps),
        opt="2site",
        nstates=int(nstates),
        verbose=int(verbose),
        not_conv_err=bool(not_conv_err),
        sweep_tol=float(sweep_tol),
        davidson_tol=float(davidson_tol),
        davidson_max_iter=int(davidson_max_iter),
        noise=float(noise),
        noise_decay=float(noise_decay),
    ).run()
    sigma_z_mpo = spin_boson_sigma_z_mpo(chain, nboson)
    magnetization = solver.ground_state.expectation(sigma_z_mpo)
    energies = np.asarray(solver.e_tot if isinstance(solver.e_tot, list) else [solver.e_tot], dtype=float)
    return SpinBosonWilsonDMRGResult(
        energies=energies,
        magnetization=float(np.real_if_close(magnetization)),
        dmrg=solver,
        mpo=mpo,
        sigma_z_mpo=sigma_z_mpo,
        dims=[2] + [int(nboson)] * chain.nmodes,
    )


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
            if getattr(result, "nrg_rescaled", False) or getattr(step, "nrg_rescaled", False):
                factor = 1.0
            else:
                factor = float(Lambda) ** (float(rescale_power) * step.site)
            row[: len(gaps)] = gaps * factor
        rows.append(row)
    return np.asarray(rows, dtype=float)


def spin_boson_narg_step_observables(
    result: SpinBosonWilsonNARGResult,
    *,
    nvalues=None,
    nlevels: int = 2,
) -> SpinBosonNARGStepObservables:
    """Extract finite-size observables from one stored max-chain NARG run.

    Step ``site`` represents the Wilson-chain prefix with ``site + 1`` boson
    modes.  Using those snapshots avoids rerunning the same sequential update
    separately for every finite-size diagnostic point.
    """
    nlevels = int(nlevels)
    if nlevels < 1:
        raise ValueError("nlevels must be positive.")

    step_by_n = {int(step.site) + 1: step for step in result.steps}
    if nvalues is None:
        requested = np.asarray(sorted(step_by_n), dtype=int)
    else:
        requested = np.asarray(nvalues, dtype=int)
        if requested.ndim != 1:
            raise ValueError("nvalues must be one-dimensional.")

    missing = [int(nmode) for nmode in requested if int(nmode) not in step_by_n]
    if missing:
        raise ValueError(
            "requested Wilson lengths are not present in the NARG result: "
            + ", ".join(str(item) for item in missing)
        )

    energies = np.full((len(requested), nlevels), np.nan, dtype=float)
    magnetizations = np.full_like(energies, np.nan)
    kept = np.zeros(len(requested), dtype=int)
    product_dims = np.zeros(len(requested), dtype=int)
    boundary_norms = np.full(len(requested), np.nan, dtype=float)
    rescale_factors = np.full(len(requested), np.nan, dtype=float)

    for row, nmode in enumerate(requested):
        step = step_by_n[int(nmode)]
        kept[row] = int(step.kept)
        product_dims[row] = int(step.product_dim)
        boundary_norms[row] = float(step.boundary_norm)
        rescale_factors[row] = float(step.rescale_factor)

        if step.energies is not None:
            values = np.asarray(step.energies, dtype=float).reshape(-1)
        elif step.effective_hamiltonian is not None:
            values = np.real_if_close(np.diag(step.effective_hamiltonian)).astype(float)
        else:
            values = np.array([], dtype=float)
        count = min(nlevels, len(values))
        if count:
            energies[row, :count] = values[:count]

        if step.sigma_z is not None:
            diag = np.real_if_close(np.diag(step.sigma_z)).astype(float)
            count = min(nlevels, len(diag))
            if count:
                magnetizations[row, :count] = diag[:count]

    return SpinBosonNARGStepObservables(
        nvalues=requested.copy(),
        energies=energies,
        gaps=energies[:, 1:] - energies[:, :1],
        magnetizations=magnetizations,
        kept=kept,
        product_dims=product_dims,
        boundary_norms=boundary_norms,
        rescale_factors=rescale_factors,
    )


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
        nrg_rescale: bool = False,
        nrg_Lambda: float = 2.0,
        nrg_rescale_power: float = 1.0,
        nrg_scale: str = "lambda",
        nrg_shift_ground: bool = True,
        diagonalization_method: str = "dense",
        sparse_diagonalization_threshold: int = 2048,
        diagonalization_tol: float = 1.0e-10,
        diagonalization_maxiter: int | None = None,
        diagonalization_ncv: int | None = None,
        initial_product_vectors=None,
        store_step_vectors: bool = False,
    ):
        self.chain = chain
        self.nboson = int(nboson)
        self.bond_dim = int(bond_dim)
        self.basis = str(basis).lower()
        self.displacements = displacements
        self.parent_dim = parent_dim
        self.symmetric_displacement = bool(symmetric_displacement)
        self.dvr_qmax = float(dvr_qmax)
        self.nrg_rescale = bool(nrg_rescale)
        self.nrg_Lambda = float(nrg_Lambda)
        self.nrg_rescale_power = float(nrg_rescale_power)
        self.nrg_scale = str(nrg_scale).lower()
        self.nrg_shift_ground = bool(nrg_shift_ground)
        self.diagonalization_method = str(diagonalization_method).lower()
        self.sparse_diagonalization_threshold = int(sparse_diagonalization_threshold)
        self.diagonalization_tol = float(diagonalization_tol)
        self.diagonalization_maxiter = (
            None if diagonalization_maxiter is None else int(diagonalization_maxiter)
        )
        self.diagonalization_ncv = None if diagonalization_ncv is None else int(diagonalization_ncv)
        self.initial_product_vectors = initial_product_vectors
        self.store_step_vectors = bool(store_step_vectors)
        if self.nboson < 1:
            raise ValueError("nboson must be positive.")
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")
        if self.nrg_Lambda <= 1.0:
            raise ValueError("nrg_Lambda must be larger than one.")
        if self.nrg_scale not in {"lambda", "onsite"}:
            raise ValueError("nrg_scale must be 'lambda' or 'onsite'.")
        if self.diagonalization_method not in {"dense", "iterative", "lobpcg", "auto"}:
            raise ValueError("diagonalization_method must be 'dense', 'iterative', 'lobpcg', or 'auto'.")
        if self.sparse_diagonalization_threshold < 1:
            raise ValueError("sparse_diagonalization_threshold must be positive.")
        if self.diagonalization_tol <= 0.0:
            raise ValueError("diagonalization_tol must be positive.")
        if self.diagonalization_maxiter is not None and self.diagonalization_maxiter < 1:
            raise ValueError("diagonalization_maxiter must be positive when supplied.")
        if self.diagonalization_ncv is not None and self.diagonalization_ncv < 1:
            raise ValueError("diagonalization_ncv must be positive when supplied.")
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
    def _diagonalize(
        hamiltonian,
        keep,
        *,
        method: str = "dense",
        sparse_threshold: int = 2048,
        tol: float = 1.0e-10,
        maxiter: int | None = None,
        ncv: int | None = None,
        initial_vectors=None,
    ):
        keep = min(int(keep), hamiltonian.shape[0])
        hermitian = 0.5 * (hamiltonian + hamiltonian.T.conj())
        dim = hermitian.shape[0]
        method = str(method).lower()
        if method not in {"dense", "iterative", "lobpcg", "auto"}:
            raise ValueError("method must be 'dense', 'iterative', 'lobpcg', or 'auto'.")
        if method == "lobpcg":
            use_iterative = False
        else:
            use_iterative = (
                method == "iterative"
                or (method == "auto" and dim >= int(sparse_threshold) and keep < dim - 1)
            )
        if use_iterative:
            if keep >= dim - 1:
                use_iterative = False
            else:
                operator = LinearOperator(
                    hermitian.shape,
                    matvec=lambda vector: hermitian @ vector,
                    dtype=np.result_type(hermitian.dtype, np.complex128),
                )
                try:
                    values, vectors = eigsh(
                        operator,
                        k=keep,
                        which="SA",
                        tol=float(tol),
                        maxiter=maxiter,
                        ncv=SpinBosonWilsonNARG._eigsh_ncv(dim, keep, ncv),
                    )
                    order = np.argsort(values.real)
                    return values[order].real, vectors[:, order]
                except ArpackNoConvergence:
                    if method == "iterative":
                        raise
        if keep < hermitian.shape[0]:
            values, vectors = eigh(
                hermitian,
                subset_by_index=(0, keep - 1),
                check_finite=False,
            )
            return values, vectors
        values, vectors = np.linalg.eigh(hermitian)
        return values, vectors

    @staticmethod
    def _eigsh_ncv(dim: int, keep: int, ncv: int | None):
        if ncv is not None:
            return min(int(dim), max(int(keep) + 1, int(ncv)))
        return min(int(dim), max(20, 4 * int(keep) + 1))

    @staticmethod
    def _orthonormalize_trial_vectors(candidates, dim: int, keep: int):
        arrays = []
        for candidate in candidates:
            if candidate is None:
                continue
            array = np.asarray(candidate, dtype=complex)
            if array.ndim != 2 or array.shape[0] != int(dim) or array.shape[1] == 0:
                continue
            arrays.append(array[:, : min(array.shape[1], int(dim))])

        rng = np.random.default_rng(12345)
        if arrays:
            matrix = np.concatenate(arrays, axis=1)
        else:
            matrix = np.empty((int(dim), 0), dtype=complex)

        while matrix.shape[1] < int(keep):
            extra = rng.normal(size=(int(dim), int(keep) - matrix.shape[1]))
            extra = extra + 1.0j * rng.normal(size=extra.shape)
            matrix = np.concatenate((matrix, extra), axis=1)

        q, r = np.linalg.qr(matrix, mode="reduced")
        independent = np.abs(np.diag(r)) > 1.0e-12
        q = q[:, independent]
        while q.shape[1] < int(keep):
            extra = rng.normal(size=(int(dim), int(keep) - q.shape[1]))
            extra = extra + 1.0j * rng.normal(size=extra.shape)
            if q.shape[1]:
                extra -= q @ (q.conj().T @ extra)
            extra_q, extra_r = np.linalg.qr(extra, mode="reduced")
            keep_extra = np.abs(np.diag(extra_r)) > 1.0e-12
            if not np.any(keep_extra):
                break
            q = np.concatenate((q, extra_q[:, keep_extra]), axis=1)
        return q[:, : int(keep)]

    @staticmethod
    def _uncoupled_product_trial_vectors(block_hamiltonian, site_hamiltonian, keep):
        block_hamiltonian = 0.5 * (block_hamiltonian + block_hamiltonian.T.conj())
        site_hamiltonian = 0.5 * (site_hamiltonian + site_hamiltonian.T.conj())
        block_values, block_vectors = eigh(block_hamiltonian, check_finite=False)
        site_values, site_vectors = eigh(site_hamiltonian, check_finite=False)
        block_dim = block_hamiltonian.shape[0]
        site_dim = site_hamiltonian.shape[0]
        dim = block_dim * site_dim
        order = np.argsort((block_values[:, None] + site_values[None, :]).reshape(-1))
        columns = np.empty((dim, min(int(keep), dim)), dtype=complex)
        for column, flat_index in enumerate(order[: columns.shape[1]]):
            block_index = int(flat_index) // site_dim
            site_index = int(flat_index) % site_dim
            columns[:, column] = np.kron(block_vectors[:, block_index], site_vectors[:, site_index])
        return columns

    @staticmethod
    def _product_eigsh_v0(block_hamiltonian, site_hamiltonian, initial_vectors=None):
        dim = block_hamiltonian.shape[0] * site_hamiltonian.shape[0]
        if initial_vectors is not None:
            candidate = np.asarray(initial_vectors, dtype=complex)
            if candidate.ndim == 2 and candidate.shape[0] == dim and candidate.shape[1]:
                vector = candidate[:, 0].copy()
                norm = np.linalg.norm(vector)
                if norm > 1.0e-14:
                    return vector / norm

        block_hamiltonian = 0.5 * (block_hamiltonian + block_hamiltonian.T.conj())
        site_hamiltonian = 0.5 * (site_hamiltonian + site_hamiltonian.T.conj())
        _block_values, block_vectors = eigh(
            block_hamiltonian,
            subset_by_index=(0, 0),
            check_finite=False,
        )
        _site_values, site_vectors = eigh(
            site_hamiltonian,
            subset_by_index=(0, 0),
            check_finite=False,
        )
        vector = np.kron(block_vectors[:, 0], site_vectors[:, 0])
        norm = np.linalg.norm(vector)
        return None if norm <= 1.0e-14 else vector / norm

    @staticmethod
    def _product_operator(block_hamiltonian, site_hamiltonian, coupling_terms):
        block_hamiltonian = np.asarray(block_hamiltonian, dtype=complex)
        site_hamiltonian = np.asarray(site_hamiltonian, dtype=complex)
        terms = [
            (np.asarray(left, dtype=complex), np.asarray(right, dtype=complex))
            for left, right in coupling_terms
        ]
        block_dim = block_hamiltonian.shape[0]
        site_dim = site_hamiltonian.shape[0]
        dim = block_dim * site_dim

        def matmat(matrix):
            columns = matrix.shape[1]
            x = np.asarray(matrix).reshape(block_dim, site_dim, columns)
            y = np.einsum("ij,jak->iak", block_hamiltonian, x, optimize=True)
            y += np.einsum("ibk,ab->iak", x, site_hamiltonian, optimize=True)
            for left, right in terms:
                y += np.einsum("ij,jbk,ab->iak", left, x, right, optimize=True)
            return y.reshape(dim, columns)

        def matvec(vector):
            return matmat(np.asarray(vector).reshape(dim, 1))[:, 0]

        return LinearOperator((dim, dim), matvec=matvec, matmat=matmat, dtype=complex)

    @staticmethod
    def _product_jacobi_preconditioner(block_hamiltonian, site_hamiltonian, coupling_terms):
        block_diag = np.asarray(np.diag(block_hamiltonian), dtype=complex)
        site_diag = np.asarray(np.diag(site_hamiltonian), dtype=complex)
        block_dim = block_hamiltonian.shape[0]
        site_dim = site_hamiltonian.shape[0]
        diagonal = np.repeat(block_diag, site_dim) + np.tile(site_diag, block_dim)
        for left, right in coupling_terms:
            left_diag = np.asarray(np.diag(left), dtype=complex)
            right_diag = np.asarray(np.diag(right), dtype=complex)
            diagonal += np.repeat(left_diag, site_dim) * np.tile(right_diag, block_dim)
        diagonal = np.real_if_close(diagonal).real

        spread = max(float(np.ptp(diagonal)), 1.0)
        shift = float(np.min(diagonal)) - spread
        inverse = 1.0 / np.maximum(diagonal - shift, 1.0e-12 * spread)
        dim = block_dim * site_dim

        def matvec(vector):
            vector = np.asarray(vector)
            if vector.ndim == 2:
                return inverse[:, None] * vector
            return inverse * vector

        def matmat(matrix):
            return inverse[:, None] * matrix

        return LinearOperator((dim, dim), matvec=matvec, matmat=matmat, dtype=complex)

    @staticmethod
    def _dense_product_hamiltonian(block_hamiltonian, site_hamiltonian, coupling_terms):
        block_dim = block_hamiltonian.shape[0]
        site_dim = site_hamiltonian.shape[0]
        hamiltonian = np.kron(block_hamiltonian, np.eye(site_dim, dtype=complex))
        hamiltonian += np.kron(np.eye(block_dim, dtype=complex), site_hamiltonian)
        for left, right in coupling_terms:
            hamiltonian += np.kron(left, right)
        return 0.5 * (hamiltonian + hamiltonian.T.conj())

    @staticmethod
    def _diagonalize_product(
        block_hamiltonian,
        site_hamiltonian,
        coupling_terms,
        keep,
        *,
        method: str = "dense",
        sparse_threshold: int = 2048,
        tol: float = 1.0e-10,
        maxiter: int | None = None,
        ncv: int | None = None,
        initial_vectors=None,
    ):
        block_hamiltonian = np.asarray(block_hamiltonian, dtype=complex)
        site_hamiltonian = np.asarray(site_hamiltonian, dtype=complex)
        coupling_terms = [
            (np.asarray(left, dtype=complex), np.asarray(right, dtype=complex))
            for left, right in coupling_terms
        ]
        dim = block_hamiltonian.shape[0] * site_hamiltonian.shape[0]
        keep = min(int(keep), dim)
        method = str(method).lower()
        if method not in {"dense", "iterative", "lobpcg", "auto"}:
            raise ValueError("method must be 'dense', 'iterative', 'lobpcg', or 'auto'.")
        if method == "lobpcg" and keep < dim - 1:
            operator = SpinBosonWilsonNARG._product_operator(
                block_hamiltonian,
                site_hamiltonian,
                coupling_terms,
            )
            trial = SpinBosonWilsonNARG._orthonormalize_trial_vectors(
                (
                    initial_vectors,
                    SpinBosonWilsonNARG._uncoupled_product_trial_vectors(
                        block_hamiltonian,
                        site_hamiltonian,
                        keep,
                    ),
                ),
                dim,
                keep,
            )
            values, vectors = lobpcg(
                operator,
                trial,
                M=SpinBosonWilsonNARG._product_jacobi_preconditioner(
                    block_hamiltonian,
                    site_hamiltonian,
                    coupling_terms,
                ),
                tol=float(tol),
                maxiter=80 if maxiter is None else int(maxiter),
                largest=False,
            )
            order = np.argsort(values.real)
            return values[order].real, vectors[:, order]

        use_iterative = (
            method == "iterative"
            or (method == "auto" and dim >= int(sparse_threshold) and keep < dim - 1)
        )
        if use_iterative and keep < dim - 1:
            operator = SpinBosonWilsonNARG._product_operator(
                block_hamiltonian,
                site_hamiltonian,
                coupling_terms,
            )
            try:
                values, vectors = eigsh(
                    operator,
                    k=keep,
                    which="SA",
                    tol=float(tol),
                    maxiter=maxiter,
                    ncv=SpinBosonWilsonNARG._eigsh_ncv(dim, keep, ncv),
                    v0=SpinBosonWilsonNARG._product_eigsh_v0(
                        block_hamiltonian,
                        site_hamiltonian,
                        initial_vectors,
                    ),
                )
                order = np.argsort(values.real)
                return values[order].real, vectors[:, order]
            except ArpackNoConvergence:
                if method == "iterative":
                    raise

        hamiltonian = SpinBosonWilsonNARG._dense_product_hamiltonian(
            block_hamiltonian,
            site_hamiltonian,
            coupling_terms,
        )
        return SpinBosonWilsonNARG._diagonalize(
            hamiltonian,
            keep,
            method="dense",
            sparse_threshold=sparse_threshold,
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
        )

    def _diagonalize_step(self, hamiltonian, keep):
        return self._diagonalize(
            hamiltonian,
            keep,
            method=self.diagonalization_method,
            sparse_threshold=self.sparse_diagonalization_threshold,
            tol=self.diagonalization_tol,
            maxiter=self.diagonalization_maxiter,
            ncv=self.diagonalization_ncv,
        )

    def _diagonalize_product_step(
        self,
        block_hamiltonian,
        site_hamiltonian,
        coupling_terms,
        keep,
        *,
        initial_vectors=None,
    ):
        return self._diagonalize_product(
            block_hamiltonian,
            site_hamiltonian,
            coupling_terms,
            keep,
            method=self.diagonalization_method,
            sparse_threshold=self.sparse_diagonalization_threshold,
            tol=self.diagonalization_tol,
            maxiter=self.diagonalization_maxiter,
            ncv=self.diagonalization_ncv,
            initial_vectors=initial_vectors,
        )

    @staticmethod
    def _project_product_operator(vectors, block_dim: int, site_operator):
        site_operator = np.asarray(site_operator, dtype=complex)
        kept = vectors.shape[1]
        site_dim = site_operator.shape[0]
        wavefunctions = np.asarray(vectors).reshape(int(block_dim), site_dim, kept)
        return np.einsum(
            "iak,ab,ibl->kl",
            wavefunctions.conj(),
            site_operator,
            wavefunctions,
            optimize=True,
        )

    @staticmethod
    def _project_block_operator(vectors, block_operator, site_dim: int):
        block_operator = np.asarray(block_operator, dtype=complex)
        kept = vectors.shape[1]
        block_dim = block_operator.shape[0]
        wavefunctions = np.asarray(vectors).reshape(block_dim, int(site_dim), kept)
        return np.einsum(
            "iak,ij,jal->kl",
            wavefunctions.conj(),
            block_operator,
            wavefunctions,
            optimize=True,
        )

    def _initial_product_vectors_for_site(self, site: int, product_dim: int):
        vectors = self.initial_product_vectors
        if vectors is None:
            return None
        if isinstance(vectors, dict):
            candidate = vectors.get(int(site))
        else:
            try:
                candidate = vectors[int(site)]
            except (IndexError, TypeError):
                return None
        if candidate is None:
            return None
        candidate = np.asarray(candidate, dtype=complex)
        if candidate.ndim != 2 or candidate.shape[0] != int(product_dim):
            return None
        return candidate

    def _scale_for_site(self, site: int) -> float:
        if not self.nrg_rescale:
            return 1.0
        if self.nrg_scale == "onsite":
            scale = abs(float(self.chain.onsite[int(site)]))
            if scale <= 0.0:
                raise ValueError("cannot use onsite NRG scaling with a zero onsite energy.")
            return 1.0 / scale
        return float(self.nrg_Lambda) ** (float(self.nrg_rescale_power) * int(site))

    def _store_energies(self, values):
        values = np.asarray(values, dtype=float)
        if self.nrg_rescale and self.nrg_shift_ground and len(values):
            return values - float(values[0])
        return values

    def run(self, nroots: int = 1) -> SpinBosonWilsonNARGResult:
        chain = self.chain
        nroots = int(nroots)
        if chain.nmodes == 0:
            values, vectors = self._diagonalize_step(chain.impurity_hamiltonian(), nroots)
            _, _, _, z = spin_operators()
            magnetizations = np.einsum("ik,ij,jk->k", vectors.conj(), z, vectors)
            stored_values = self._store_energies(values)
            return SpinBosonWilsonNARGResult(
                stored_values,
                vectors,
                [],
                np.diag(stored_values),
                None,
                sigma_z=z,
                magnetizations=np.real_if_close(magnetizations),
                nrg_rescaled=self.nrg_rescale,
                nrg_Lambda=self.nrg_Lambda,
                nrg_rescale_power=self.nrg_rescale_power,
                nrg_scale=self.nrg_scale,
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

        current_scale = self._scale_for_site(0)
        block_hamiltonian = current_scale * chain.impurity_hamiltonian()
        site_hamiltonian = current_scale * chain.onsite[0] * number
        coupling_terms = (
            (0.5 * current_scale * chain.impurity_coupling * z, b + bdag),
        )
        product_dim = block_hamiltonian.shape[0] * identity_b.shape[0]
        keep = min(self.bond_dim, product_dim)
        initial_vectors = self._initial_product_vectors_for_site(0, product_dim)
        values, vectors = self._diagonalize_product_step(
            block_hamiltonian,
            site_hamiltonian,
            coupling_terms,
            keep,
            initial_vectors=initial_vectors,
        )
        stored_values = self._store_energies(values)
        boundary_b = self._project_product_operator(vectors, 2, b)
        sigma_z = self._project_block_operator(vectors, z, self.nboson)
        effective_hamiltonian = np.diag(stored_values).astype(complex)
        steps.append(
            SpinBosonWilsonNARGStep(
                site=0,
                product_dim=product_dim,
                kept=len(values),
                lowest_energy=float(values[0]),
                boundary_norm=float(np.linalg.norm(boundary_b)),
                energies=stored_values.copy(),
                effective_hamiltonian=effective_hamiltonian.copy(),
                boundary_annihilation=boundary_b.copy(),
                sigma_z=sigma_z.copy(),
                product_vectors=vectors.copy() if self.store_step_vectors else None,
                rescale_factor=current_scale,
                nrg_rescaled=self.nrg_rescale,
            )
        )

        for site in range(1, chain.nmodes):
            block_dim = effective_hamiltonian.shape[0]
            identity_b, b, bdag, number = site_ops[site]
            current_scale = self._scale_for_site(site)
            previous_scale = self._scale_for_site(site - 1)
            block_scale = current_scale / previous_scale
            shell_scale = current_scale
            block_hamiltonian = block_scale * effective_hamiltonian
            site_hamiltonian = shell_scale * chain.onsite[site] * number
            coupling_terms = (
                (shell_scale * chain.hopping[site - 1] * boundary_b.conj().T, b),
                (shell_scale * chain.hopping[site - 1] * boundary_b, bdag),
            )
            product_dim = block_dim * identity_b.shape[0]
            keep = min(self.bond_dim, product_dim)
            initial_vectors = self._initial_product_vectors_for_site(site, product_dim)
            values, vectors = self._diagonalize_product_step(
                block_hamiltonian,
                site_hamiltonian,
                coupling_terms,
                keep,
                initial_vectors=initial_vectors,
            )
            stored_values = self._store_energies(values)
            boundary_b = self._project_product_operator(vectors, block_dim, b)
            sigma_z = self._project_block_operator(vectors, sigma_z, self.nboson)
            effective_hamiltonian = np.diag(stored_values).astype(complex)
            steps.append(
                SpinBosonWilsonNARGStep(
                    site=site,
                    product_dim=product_dim,
                    kept=len(values),
                    lowest_energy=float(values[0]),
                    boundary_norm=float(np.linalg.norm(boundary_b)),
                    energies=stored_values.copy(),
                    effective_hamiltonian=effective_hamiltonian.copy(),
                    boundary_annihilation=boundary_b.copy(),
                    sigma_z=sigma_z.copy(),
                    product_vectors=vectors.copy() if self.store_step_vectors else None,
                    rescale_factor=current_scale,
                    nrg_rescaled=self.nrg_rescale,
                )
            )

        magnetizations = np.diag(sigma_z)[:nroots]
        return SpinBosonWilsonNARGResult(
            energies=stored_values[:nroots],
            vectors=vectors[:, :nroots],
            steps=steps,
            effective_hamiltonian=effective_hamiltonian,
            boundary_annihilation=boundary_b,
            sigma_z=sigma_z,
            magnetizations=np.real_if_close(magnetizations),
            nrg_rescaled=self.nrg_rescale,
            nrg_Lambda=self.nrg_Lambda,
            nrg_rescale_power=self.nrg_rescale_power,
            nrg_scale=self.nrg_scale,
        )


def _phase_align_columns(vectors: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Align conditional eigenvector phases without mixing eigenstates."""
    if reference is None:
        return vectors
    count = min(vectors.shape[1], reference.shape[1])
    out = vectors.copy()
    for column in range(count):
        overlap = np.vdot(reference[:, column], out[:, column])
        if abs(overlap) > 1e-14:
            out[:, column] *= np.conj(overlap) / abs(overlap)
    return out


def _conditional_site_hamiltonian(
    *,
    block_hamiltonian: np.ndarray,
    sigma_z_block: np.ndarray,
    x_operator: np.ndarray,
    p_operator: np.ndarray | None,
    onsite_frequency: float,
    q_grid: np.ndarray,
    kinetic: np.ndarray,
    momentum: np.ndarray,
    n_conditional_states: int,
):
    """Build one orthonormal DVR conditional-basis NARG Hamiltonian.

    The conditional basis is ``|q_i> |A_a(q_i)>`` with

    ``[H_block + q_i X_block + omega*(q_i**2/2 - 1/2)] |A_a(q_i)> = eps_ia |A_a(q_i)>``.

    Since ``<q_i|q_j> = delta_ij``, the total overlap matrix is the identity.
    The slow kinetic and Wilson momentum-hopping terms become dressed matrix
    elements between conditional frames.
    """
    block_hamiltonian = np.asarray(block_hamiltonian, dtype=complex)
    sigma_z_block = np.asarray(sigma_z_block, dtype=complex)
    x_operator = np.asarray(x_operator, dtype=complex)
    p_operator = None if p_operator is None else np.asarray(p_operator, dtype=complex)
    q_grid = np.asarray(q_grid, dtype=float)
    kinetic = np.asarray(kinetic, dtype=complex)
    momentum = np.asarray(momentum, dtype=complex)

    block_dim = block_hamiltonian.shape[0]
    n_conditional_states = min(int(n_conditional_states), block_dim)
    if n_conditional_states < 1:
        raise ValueError("n_conditional_states must be positive.")

    frames = np.empty((len(q_grid), block_dim, n_conditional_states), dtype=complex)
    conditional_energies = np.empty((len(q_grid), n_conditional_states), dtype=float)
    identity_block = np.eye(block_dim, dtype=complex)
    previous = None
    for index, coordinate in enumerate(q_grid):
        potential = float(onsite_frequency) * (0.5 * coordinate**2 - 0.5)
        conditional = block_hamiltonian + coordinate * x_operator + potential * identity_block
        values, vectors = SpinBosonWilsonNARG._diagonalize(conditional, n_conditional_states)
        vectors = _phase_align_columns(vectors, previous)
        conditional_energies[index] = values[:n_conditional_states]
        frames[index] = vectors[:, :n_conditional_states]
        previous = frames[index]

    ngrid = len(q_grid)
    dim = ngrid * n_conditional_states
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    local_b = np.zeros_like(hamiltonian)
    sigma_z = np.zeros_like(hamiltonian)

    for i in range(ngrid):
        row = slice(i * n_conditional_states, (i + 1) * n_conditional_states)
        hamiltonian[row, row] += np.diag(conditional_energies[i])
        local_b[row, row] += q_grid[i] / np.sqrt(2.0) * np.eye(n_conditional_states)
        sigma_z[row, row] += frames[i].conj().T @ sigma_z_block @ frames[i]

        for j in range(ngrid):
            col = slice(j * n_conditional_states, (j + 1) * n_conditional_states)
            overlap = frames[i].conj().T @ frames[j]
            if abs(kinetic[i, j]) > 1e-14:
                hamiltonian[row, col] += onsite_frequency * kinetic[i, j] * overlap
            if abs(momentum[i, j]) > 1e-14:
                local_b[row, col] += 1.0j * momentum[i, j] / np.sqrt(2.0) * overlap
                if p_operator is not None:
                    hamiltonian[row, col] += momentum[i, j] * (frames[i].conj().T @ p_operator @ frames[j])

    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T.conj())
    sigma_z = 0.5 * (sigma_z + sigma_z.T.conj())
    return hamiltonian, local_b, sigma_z, conditional_energies, frames


class SpinBosonWilsonAdiabaticNARG(SpinBosonWilsonNARG):
    """Orthogonal conditional-basis NARG for spin-boson Wilson chains.

    This is the NARG/Born-Huang version of the Wilson-chain growth step.  The
    new oscillator is represented on a sine-DVR coordinate grid, and the block
    is diagonalized conditionally at each grid point.  The total basis remains
    orthonormal, so no generalized overlap matrix is solved; nonadiabatic
    effects appear as overlap-dressed kinetic and momentum-hopping terms.
    """

    def __init__(
        self,
        chain: SpinBosonWilsonChain,
        *,
        nboson: int = 16,
        bond_dim: int = 32,
        n_conditional_states: int | None = None,
        dvr_qmax: float = 8.0,
        nrg_rescale: bool = False,
        nrg_Lambda: float = 2.0,
        nrg_rescale_power: float = 1.0,
        nrg_scale: str = "lambda",
        nrg_shift_ground: bool = True,
        diagonalization_method: str = "auto",
        sparse_diagonalization_threshold: int = 2048,
        diagonalization_tol: float = 1.0e-10,
        diagonalization_maxiter: int | None = None,
        diagonalization_ncv: int | None = None,
        initial_product_vectors=None,
        store_step_vectors: bool = False,
        full_conditional_shortcut: bool = True,
    ):
        super().__init__(
            chain,
            nboson=nboson,
            bond_dim=bond_dim,
            basis="sine-dvr",
            dvr_qmax=dvr_qmax,
            nrg_rescale=nrg_rescale,
            nrg_Lambda=nrg_Lambda,
            nrg_rescale_power=nrg_rescale_power,
            nrg_scale=nrg_scale,
            nrg_shift_ground=nrg_shift_ground,
            diagonalization_method=diagonalization_method,
            sparse_diagonalization_threshold=sparse_diagonalization_threshold,
            diagonalization_tol=diagonalization_tol,
            diagonalization_maxiter=diagonalization_maxiter,
            diagonalization_ncv=diagonalization_ncv,
            initial_product_vectors=initial_product_vectors,
            store_step_vectors=store_step_vectors,
        )
        self.n_conditional_states = n_conditional_states
        self.full_conditional_shortcut = bool(full_conditional_shortcut)

    def _conditional_count(self, block_dim: int) -> int:
        if self.n_conditional_states is None:
            return int(block_dim)
        return min(int(self.n_conditional_states), int(block_dim))

    def _uses_full_conditional_basis(self) -> bool:
        return self.n_conditional_states is None or int(self.n_conditional_states) >= self.bond_dim

    def _site_coordinate_operators(self):
        return [
            sine_dvr_boson_operators(
                self.nboson,
                qmax=self.dvr_qmax,
            )
            for _ in range(self.chain.nmodes)
        ]

    def _run_full_conditional_shortcut(self, nroots: int) -> SpinBosonWilsonAdiabaticNARGResult:
        """Use the product sine-DVR update equivalent to full conditional NARG."""
        base = SpinBosonWilsonNARG(
            self.chain,
            nboson=self.nboson,
            bond_dim=self.bond_dim,
            basis="sine-dvr",
            dvr_qmax=self.dvr_qmax,
            nrg_rescale=self.nrg_rescale,
            nrg_Lambda=self.nrg_Lambda,
            nrg_rescale_power=self.nrg_rescale_power,
            nrg_scale=self.nrg_scale,
            nrg_shift_ground=self.nrg_shift_ground,
            diagonalization_method=self.diagonalization_method,
            sparse_diagonalization_threshold=self.sparse_diagonalization_threshold,
            diagonalization_tol=self.diagonalization_tol,
            diagonalization_maxiter=self.diagonalization_maxiter,
            diagonalization_ncv=self.diagonalization_ncv,
            initial_product_vectors=self.initial_product_vectors,
            store_step_vectors=self.store_step_vectors,
        ).run(nroots=nroots)

        site_ops = self._site_coordinate_operators()
        converted_steps = []
        for index, step in enumerate(base.steps):
            conditional_dim = 2 if index == 0 else int(base.steps[index - 1].kept)
            q_grid = site_ops[index][4] if index < len(site_ops) else None
            converted_steps.append(
                SpinBosonWilsonAdiabaticNARGStep(
                    site=step.site,
                    product_dim=step.product_dim,
                    kept=step.kept,
                    lowest_energy=step.lowest_energy,
                    boundary_norm=step.boundary_norm,
                    energies=None if step.energies is None else step.energies.copy(),
                    effective_hamiltonian=(
                        None if step.effective_hamiltonian is None else step.effective_hamiltonian.copy()
                    ),
                    boundary_annihilation=(
                        None if step.boundary_annihilation is None else step.boundary_annihilation.copy()
                    ),
                    sigma_z=None if step.sigma_z is None else step.sigma_z.copy(),
                    product_vectors=(
                        None if step.product_vectors is None else step.product_vectors.copy()
                    ),
                    rescale_factor=step.rescale_factor,
                    nrg_rescaled=step.nrg_rescaled,
                    conditional_dim=conditional_dim,
                    q_grid=None if q_grid is None else q_grid.copy(),
                    conditional_energies=None,
                )
            )

        return SpinBosonWilsonAdiabaticNARGResult(
            energies=base.energies,
            vectors=base.vectors,
            steps=converted_steps,
            effective_hamiltonian=base.effective_hamiltonian,
            boundary_annihilation=base.boundary_annihilation,
            sigma_z=base.sigma_z,
            magnetizations=base.magnetizations,
            nrg_rescaled=base.nrg_rescaled,
            nrg_Lambda=base.nrg_Lambda,
            nrg_rescale_power=base.nrg_rescale_power,
            nrg_scale=base.nrg_scale,
        )

    def run(self, nroots: int = 1) -> SpinBosonWilsonAdiabaticNARGResult:
        chain = self.chain
        nroots = int(nroots)
        if chain.nmodes == 0:
            base = super().run(nroots=nroots)
            return SpinBosonWilsonAdiabaticNARGResult(**base.__dict__)
        if self.full_conditional_shortcut and self._uses_full_conditional_basis():
            return self._run_full_conditional_shortcut(nroots=nroots)

        site_ops = self._site_coordinate_operators()
        _, _, _, z = spin_operators()
        steps = []

        _, _, _, _, q_grid, kinetic, momentum = site_ops[0]
        current_scale = self._scale_for_site(0)
        hamiltonian, local_b, sigma_z_full, conditional_energies, _ = _conditional_site_hamiltonian(
            block_hamiltonian=current_scale * chain.impurity_hamiltonian(),
            sigma_z_block=z,
            x_operator=current_scale * chain.impurity_coupling / np.sqrt(2.0) * z,
            p_operator=None,
            onsite_frequency=current_scale * chain.onsite[0],
            q_grid=q_grid,
            kinetic=kinetic,
            momentum=momentum,
            n_conditional_states=self._conditional_count(2),
        )
        keep = min(self.bond_dim, hamiltonian.shape[0])
        values, vectors = self._diagonalize_step(hamiltonian, keep)
        stored_values = self._store_energies(values)
        boundary_b = vectors.conj().T @ local_b @ vectors
        sigma_z = vectors.conj().T @ sigma_z_full @ vectors
        effective_hamiltonian = np.diag(stored_values).astype(complex)
        steps.append(
            SpinBosonWilsonAdiabaticNARGStep(
                site=0,
                product_dim=hamiltonian.shape[0],
                kept=len(values),
                lowest_energy=float(values[0]),
                boundary_norm=float(np.linalg.norm(boundary_b)),
                energies=stored_values.copy(),
                effective_hamiltonian=effective_hamiltonian.copy(),
                boundary_annihilation=boundary_b.copy(),
                sigma_z=sigma_z.copy(),
                rescale_factor=current_scale,
                nrg_rescaled=self.nrg_rescale,
                conditional_dim=self._conditional_count(2),
                q_grid=q_grid.copy(),
                conditional_energies=conditional_energies.copy(),
            )
        )

        for site in range(1, chain.nmodes):
            block_dim = effective_hamiltonian.shape[0]
            _, _, _, _, q_grid, kinetic, momentum = site_ops[site]
            current_scale = self._scale_for_site(site)
            previous_scale = self._scale_for_site(site - 1)
            block_scale = current_scale / previous_scale
            shell_scale = current_scale
            boundary_x = (boundary_b + boundary_b.T.conj()) / np.sqrt(2.0)
            boundary_p = 1.0j * (boundary_b.T.conj() - boundary_b) / np.sqrt(2.0)
            hamiltonian, local_b, sigma_z_full, conditional_energies, _ = _conditional_site_hamiltonian(
                block_hamiltonian=block_scale * effective_hamiltonian,
                sigma_z_block=sigma_z,
                x_operator=shell_scale * chain.hopping[site - 1] * boundary_x,
                p_operator=shell_scale * chain.hopping[site - 1] * boundary_p,
                onsite_frequency=shell_scale * chain.onsite[site],
                q_grid=q_grid,
                kinetic=kinetic,
                momentum=momentum,
                n_conditional_states=self._conditional_count(block_dim),
            )
            keep = min(self.bond_dim, hamiltonian.shape[0])
            values, vectors = self._diagonalize_step(hamiltonian, keep)
            stored_values = self._store_energies(values)
            boundary_b = vectors.conj().T @ local_b @ vectors
            sigma_z = vectors.conj().T @ sigma_z_full @ vectors
            effective_hamiltonian = np.diag(stored_values).astype(complex)
            steps.append(
                SpinBosonWilsonAdiabaticNARGStep(
                    site=site,
                    product_dim=hamiltonian.shape[0],
                    kept=len(values),
                    lowest_energy=float(values[0]),
                    boundary_norm=float(np.linalg.norm(boundary_b)),
                    energies=stored_values.copy(),
                    effective_hamiltonian=effective_hamiltonian.copy(),
                    boundary_annihilation=boundary_b.copy(),
                    sigma_z=sigma_z.copy(),
                    rescale_factor=current_scale,
                    nrg_rescaled=self.nrg_rescale,
                    conditional_dim=self._conditional_count(block_dim),
                    q_grid=q_grid.copy(),
                    conditional_energies=conditional_energies.copy(),
                )
            )

        magnetizations = np.diag(sigma_z)[:nroots]
        return SpinBosonWilsonAdiabaticNARGResult(
            energies=stored_values[:nroots],
            vectors=vectors[:, :nroots],
            steps=steps,
            effective_hamiltonian=effective_hamiltonian,
            boundary_annihilation=boundary_b,
            sigma_z=sigma_z,
            magnetizations=np.real_if_close(magnetizations),
            nrg_rescaled=self.nrg_rescale,
            nrg_Lambda=self.nrg_Lambda,
            nrg_rescale_power=self.nrg_rescale_power,
            nrg_scale=self.nrg_scale,
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
    "SpinBosonNARGStepObservables",
    "SpinBosonWilsonAdiabaticNARG",
    "SpinBosonWilsonAdiabaticNARGResult",
    "SpinBosonWilsonAdiabaticNARGStep",
    "SpinBosonWilsonDMRGResult",
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
    "spin_boson_narg_step_observables",
    "scan_spin_boson_fixed_point",
    "scan_spin_boson_fixed_point_flows",
    "scan_spin_boson_fpes_alpha",
    "scan_spin_boson_fpes_observables",
    "scan_spin_boson_gap_thresholds",
    "scan_spin_boson_alpha",
    "sine_dvr_boson_operators",
    "spin_boson_mode_pes",
    "spin_boson_product_mps",
    "spin_boson_sigma_z_mpo",
    "spin_boson_wilson_dmrg",
    "spin_boson_wilson_exact",
    "spin_boson_wilson_exact_magnetization",
    "spin_boson_wilson_hamiltonian",
    "spin_boson_wilson_mpo",
    "spin_operators",
    "star_to_wilson_chain",
]
