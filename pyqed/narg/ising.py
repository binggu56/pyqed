"""Critical Ising utilities for scale-invariant NARG experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh


def pauli_matrices():
    """Return I, X, Y, Z Pauli matrices."""
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return identity, x, y, z


def _many_body_operator(local_ops, *, sparse=True):
    if not local_ops:
        raise ValueError("local_ops must be non-empty.")
    out = csr_matrix(local_ops[0]) if sparse else np.asarray(local_ops[0])
    for op in local_ops[1:]:
        out = kron(out, csr_matrix(op), format="csr") if sparse else np.kron(out, op)
    return out


def transverse_field_ising_hamiltonian(
    nsites: int,
    *,
    j: float = 1.0,
    field: float = 1.0,
    periodic: bool = True,
    sparse: bool = True,
):
    """Build ``-j sum Z_i Z_j - field sum X_i`` for a spin-1/2 chain."""
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    identity, x, _, z = pauli_matrices()
    ops_identity = [identity] * nsites
    dim = 2**nsites
    hamiltonian = csr_matrix((dim, dim), dtype=complex) if sparse else np.zeros((dim, dim), dtype=complex)

    for site in range(nsites):
        ops = list(ops_identity)
        ops[site] = x
        hamiltonian = hamiltonian - field * _many_body_operator(ops, sparse=sparse)

    bonds = [(site, site + 1) for site in range(nsites - 1)]
    if periodic and nsites > 2:
        bonds.append((nsites - 1, 0))
    for left, right in bonds:
        ops = list(ops_identity)
        ops[left] = z
        ops[right] = z
        hamiltonian = hamiltonian - j * _many_body_operator(ops, sparse=sparse)

    return hamiltonian


@dataclass
class IsingFiniteSizeScaling:
    """Finite-size scaling dimensions extracted from the critical Ising spectrum."""

    nsites: int
    energies: np.ndarray
    gaps: np.ndarray
    dimensions: np.ndarray
    velocity: float


def finite_size_scaling_dimensions(
    nsites: int,
    *,
    j: float = 1.0,
    field: float = 1.0,
    nlevels: int = 8,
    periodic: bool = True,
) -> IsingFiniteSizeScaling:
    """Estimate critical Ising scaling dimensions from finite-size gaps.

    At the critical point ``field=j`` and with periodic boundaries,
    ``Delta = L (E_n - E_0) / (2 pi v)`` with velocity ``v=2j``.
    """
    hamiltonian = transverse_field_ising_hamiltonian(
        nsites,
        j=j,
        field=field,
        periodic=periodic,
        sparse=True,
    )
    dim = hamiltonian.shape[0]
    nlevels = min(int(nlevels), dim)
    if nlevels < 1:
        raise ValueError("nlevels must be positive.")
    if dim <= max(64, nlevels + 2):
        energies = np.linalg.eigvalsh(hamiltonian.toarray())[:nlevels]
    else:
        energies = eigsh(hamiltonian, k=nlevels, which="SA", return_eigenvectors=False)
        energies = np.sort(np.real_if_close(energies))
    gaps = energies - energies[0]
    velocity = 2.0 * float(j)
    dimensions = int(nsites) * gaps / (2.0 * np.pi * velocity)
    return IsingFiniteSizeScaling(
        nsites=int(nsites),
        energies=np.asarray(energies, dtype=float),
        gaps=np.asarray(gaps, dtype=float),
        dimensions=np.asarray(dimensions, dtype=float),
        velocity=velocity,
    )


@dataclass
class IsingNARGAscending:
    """One-layer two-site NARG/Kadanoff ascending-map diagnostics."""

    isometry: np.ndarray
    block_energies: np.ndarray
    superoperator: np.ndarray
    eigenvalues: np.ndarray
    dimensions: np.ndarray
    operator_basis: tuple[str, ...]


@dataclass
class TransverseFieldIsingNARGStep:
    """One conditioned growth step in the transverse-field Ising NARG."""

    site: int
    input_dim: int
    kept: int
    branch_energies: np.ndarray
    tensor: np.ndarray
    input_symmetry_operator: np.ndarray
    output_symmetry_operator: np.ndarray


@dataclass
class TransverseFieldIsingNARGResult:
    """Sequential NARG result for the transverse-field Ising chain."""

    energies: np.ndarray
    vectors: np.ndarray
    effective_hamiltonian: np.ndarray
    symmetry_operator: np.ndarray
    steps: list[TransverseFieldIsingNARGStep]


@dataclass
class IsingNARGFixedLayerScaling:
    """Scaling dimensions from the final fixed-layer NARG tangent map."""

    superoperator: np.ndarray
    eigenvalues: np.ndarray
    dimensions: np.ndarray
    tensor: np.ndarray
    sector: str | None = None


def _last_site_z(nsites: int):
    identity, _, _, z = pauli_matrices()
    return np.kron(np.eye(2 ** (int(nsites) - 1), dtype=complex), z)


def _spin_flip_operator(nsites: int):
    _, x, _, _ = pauli_matrices()
    out = x
    for _ in range(int(nsites) - 1):
        out = np.kron(out, x)
    return out


def conditioned_spin_flip_operator(tensor: np.ndarray, input_symmetry: np.ndarray):
    """Propagate the Ising global spin-flip through a conditioned NARG tensor."""
    tensor = np.asarray(tensor, dtype=complex)
    input_symmetry = np.asarray(input_symmetry, dtype=complex)
    if tensor.ndim != 3 or tensor.shape[2] != 2:
        raise ValueError("tensor must have shape (input_dim, kept, 2).")
    if input_symmetry.shape != (tensor.shape[0], tensor.shape[0]):
        raise ValueError("input_symmetry shape must match tensor input dimension.")
    kept = tensor.shape[1]
    output = np.zeros((2 * kept, 2 * kept), dtype=complex)
    for out_branch, in_branch in ((0, 1), (1, 0)):
        block = tensor[:, :, out_branch].conj().T @ input_symmetry @ tensor[:, :, in_branch]
        output[out_branch::2, in_branch::2] = block
    return 0.5 * (output + output.T.conj())


def _conditioned_effective_hamiltonian(branch_energies, tensors, *, field: float):
    """Assemble the effective Hamiltonian in ``(kept, boundary_z)`` ordering."""
    branch_energies = np.asarray(branch_energies, dtype=float)
    tensors = np.asarray(tensors, dtype=complex)
    if branch_energies.ndim != 2 or branch_energies.shape[0] != 2:
        raise ValueError("branch_energies must have shape (2, kept).")
    if tensors.ndim != 3 or tensors.shape[1:] != (branch_energies.shape[1], 2):
        raise ValueError("tensors must have shape (input_dim, kept, 2).")
    _, x, _, _ = pauli_matrices()
    kept = branch_energies.shape[1]
    hamiltonian = np.zeros((2 * kept, 2 * kept), dtype=complex)

    for branch in range(2):
        for state in range(kept):
            hamiltonian[2 * state + branch, 2 * state + branch] = branch_energies[branch, state]

    for left_branch in range(2):
        left = tensors[:, :, left_branch]
        for right_branch in range(2):
            overlap = left.conj().T @ tensors[:, :, right_branch]
            hamiltonian[
                left_branch::2,
                right_branch::2,
            ] -= field * x[left_branch, right_branch] * overlap

    return 0.5 * (hamiltonian + hamiltonian.T.conj())


class TransverseFieldIsingNARG:
    """Conditioned sequential NARG for the transverse-field Ising chain.

    The effective basis is conditioned on the next site's ``Z`` eigenvalue.
    This follows the old exploratory TFIM NARG script, but keeps the row order
    explicit as ``(kept_state, boundary_z)``.
    """

    def __init__(
        self,
        nsites: int,
        *,
        j: float = 1.0,
        field: float = 1.0,
        bond_dim: int = 8,
        nstart: int = 2,
    ):
        self.nsites = int(nsites)
        self.j = float(j)
        self.field = float(field)
        self.bond_dim = int(bond_dim)
        self.nstart = int(nstart)
        if self.nsites < 2:
            raise ValueError("nsites must be at least 2.")
        if self.nstart < 1 or self.nstart >= self.nsites:
            raise ValueError("nstart must satisfy 1 <= nstart < nsites.")
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")

    def _diagonalize(self, hamiltonian, keep):
        keep = min(int(keep), hamiltonian.shape[0])
        values, vectors = np.linalg.eigh(0.5 * (hamiltonian + hamiltonian.T.conj()))
        return values[:keep], vectors[:, :keep]

    def _initial_step(self):
        h0 = transverse_field_ising_hamiltonian(
            self.nstart,
            j=self.j,
            field=self.field,
            periodic=False,
            sparse=False,
        )
        z_last = _last_site_z(self.nstart)
        keep = min(self.bond_dim, h0.shape[0])
        branch_energies = np.empty((2, keep), dtype=float)
        tensor = np.empty((h0.shape[0], keep, 2), dtype=complex)
        for branch, z_external in enumerate((1.0, -1.0)):
            h_branch = h0 - self.j * z_external * z_last
            branch_energies[branch], tensor[:, :, branch] = self._diagonalize(h_branch, keep)
        return branch_energies, tensor

    def run(self, nroots: int = 6) -> TransverseFieldIsingNARGResult:
        _, _, _, z = pauli_matrices()
        symmetry_operator = _spin_flip_operator(self.nstart)
        branch_energies, tensor = self._initial_step()
        input_symmetry = symmetry_operator
        hamiltonian = _conditioned_effective_hamiltonian(branch_energies, tensor, field=self.field)
        symmetry_operator = conditioned_spin_flip_operator(tensor, symmetry_operator)
        steps = [
            TransverseFieldIsingNARGStep(
                site=self.nstart,
                input_dim=tensor.shape[0],
                kept=tensor.shape[1],
                branch_energies=branch_energies.copy(),
                tensor=tensor.copy(),
                input_symmetry_operator=input_symmetry.copy(),
                output_symmetry_operator=symmetry_operator.copy(),
            )
        ]

        for site in range(self.nstart + 1, self.nsites):
            input_dim = hamiltonian.shape[0]
            input_symmetry = symmetry_operator
            keep = min(self.bond_dim, input_dim)
            branch_energies = np.empty((2, keep), dtype=float)
            tensor = np.empty((input_dim, keep, 2), dtype=complex)
            coupling = np.kron(np.eye(input_dim // 2, dtype=complex), z)
            for branch, z_external in enumerate((1.0, -1.0)):
                h_branch = hamiltonian - self.j * z_external * coupling
                branch_energies[branch], tensor[:, :, branch] = self._diagonalize(h_branch, keep)
            hamiltonian = _conditioned_effective_hamiltonian(branch_energies, tensor, field=self.field)
            symmetry_operator = conditioned_spin_flip_operator(tensor, symmetry_operator)
            steps.append(
                TransverseFieldIsingNARGStep(
                    site=site,
                    input_dim=input_dim,
                    kept=keep,
                    branch_energies=branch_energies.copy(),
                    tensor=tensor.copy(),
                    input_symmetry_operator=input_symmetry.copy(),
                    output_symmetry_operator=symmetry_operator.copy(),
                )
            )

        nroots = min(int(nroots), hamiltonian.shape[0])
        energies, vectors = self._diagonalize(hamiltonian, nroots)
        return TransverseFieldIsingNARGResult(
            energies=np.asarray(energies, dtype=float),
            vectors=vectors,
            effective_hamiltonian=hamiltonian,
            symmetry_operator=symmetry_operator,
            steps=steps,
        )


def narg_fixed_layer_superoperator(tensor: np.ndarray) -> np.ndarray:
    """Return the full operator ascending map from one real NARG layer.

    For a fixed layer ``U[:, kept, boundary_z]`` and an input operator ``O``,
    the ascended operator is block diagonal in the new boundary ``Z`` label:
    ``A(O)_(b,m),(a,m) = U_m^dag O U_m``.
    """
    tensor = np.asarray(tensor, dtype=complex)
    if tensor.ndim != 3 or tensor.shape[2] != 2:
        raise ValueError("tensor must have shape (input_dim, kept, 2).")
    input_dim, kept, _ = tensor.shape
    output_dim = 2 * kept
    if input_dim != output_dim:
        raise ValueError("fixed-layer scaling requires input_dim == 2 * kept.")
    superoperator = np.zeros((output_dim * output_dim, input_dim * input_dim), dtype=complex)
    for row in range(input_dim):
        for col in range(input_dim):
            operator = np.zeros((input_dim, input_dim), dtype=complex)
            operator[row, col] = 1.0
            ascended = np.zeros((output_dim, output_dim), dtype=complex)
            for branch in range(2):
                block = tensor[:, :, branch].conj().T @ operator @ tensor[:, :, branch]
                ascended[branch::2, branch::2] = block
            superoperator[:, row * input_dim + col] = ascended.reshape(-1)
    return superoperator


def narg_fixed_layer_scaling_dimensions(
    tensor: np.ndarray,
    *,
    block_factor: float = 2.0,
    symmetry_operator: np.ndarray | None = None,
    input_symmetry_operator: np.ndarray | None = None,
    sector: str | None = None,
) -> IsingNARGFixedLayerScaling:
    """Extract dimensions from a real NARG fixed-layer tangent map."""
    superoperator = narg_fixed_layer_superoperator(tensor)
    if sector is not None:
        if symmetry_operator is None:
            raise ValueError("symmetry_operator is required when sector is requested.")
        if input_symmetry_operator is None:
            input_symmetry_operator = symmetry_operator
        input_basis = operator_symmetry_basis(input_symmetry_operator, sector=sector)
        output_basis = operator_symmetry_basis(symmetry_operator, sector=sector)
        superoperator = output_basis.conj().T @ superoperator @ input_basis
    eigenvalues = np.linalg.eigvals(superoperator)
    with np.errstate(divide="ignore", invalid="ignore"):
        dimensions = -np.log(np.abs(eigenvalues)) / np.log(float(block_factor))
    finite = np.isfinite(dimensions)
    order = np.lexsort((np.abs(eigenvalues[finite]), dimensions[finite]))
    return IsingNARGFixedLayerScaling(
        superoperator=superoperator,
        eigenvalues=eigenvalues[finite][order],
        dimensions=dimensions[finite][order],
        tensor=np.asarray(tensor, dtype=complex),
        sector=sector,
    )


def operator_symmetry_basis(symmetry_operator: np.ndarray, *, sector: str):
    """Return an orthonormal vectorized operator basis for a Z2 sector.

    ``sector="even"`` keeps operators satisfying ``P O P = O`` and
    ``sector="odd"`` keeps operators satisfying ``P O P = -O``.
    """
    symmetry_operator = np.asarray(symmetry_operator, dtype=complex)
    if symmetry_operator.ndim != 2 or symmetry_operator.shape[0] != symmetry_operator.shape[1]:
        raise ValueError("symmetry_operator must be a square matrix.")
    dim = symmetry_operator.shape[0]
    sector_key = str(sector).lower()
    if sector_key not in {"even", "odd"}:
        raise ValueError("sector must be 'even' or 'odd'.")
    target = 1.0 if sector_key == "even" else -1.0
    conjugation = np.empty((dim * dim, dim * dim), dtype=complex)
    for row in range(dim):
        for col in range(dim):
            operator = np.zeros((dim, dim), dtype=complex)
            operator[row, col] = 1.0
            transformed = symmetry_operator @ operator @ symmetry_operator.conj().T
            conjugation[:, row * dim + col] = transformed.reshape(-1)
    values, vectors = np.linalg.eigh(0.5 * (conjugation + conjugation.T.conj()))
    selected = np.abs(values - target) < 1e-8
    if not np.any(selected):
        raise ValueError(f"no operator basis vectors found for {sector!r} sector.")
    basis, _ = np.linalg.qr(vectors[:, selected])
    return basis


def two_site_ising_isometry(
    *,
    j: float = 1.0,
    field: float = 1.0,
    keep: int = 2,
    field_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the low-energy two-site isometry used by the toy NARG layer."""
    keep = int(keep)
    if keep < 1 or keep > 4:
        raise ValueError("keep must be between 1 and 4.")
    identity, x, _, z = pauli_matrices()
    hamiltonian = (
        -j * np.kron(z, z)
        - field_weight * field * (np.kron(x, identity) + np.kron(identity, x))
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    order = np.argsort(energies)
    return vectors[:, order[:keep]], energies[order[:keep]]


def narg_ascending_superoperator(
    isometry: np.ndarray,
    *,
    average_support: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the one-site ascending map in the Pauli traceless basis.

    This is a minimal scale-invariant NARG diagnostic: the same two-site
    isometry is assumed to be reused at every layer.  It is intentionally a
    Kadanoff-style baseline, not an optimized MERA layer.
    """
    isometry = np.asarray(isometry, dtype=complex)
    if isometry.shape[0] != 4:
        raise ValueError("isometry must map a two-spin space, so shape[0] must be 4.")
    identity, x, y, z = pauli_matrices()
    basis = (x, y, z)
    names = ("X", "Y", "Z")
    superoperator = np.empty((3, 3), dtype=complex)
    for col, op in enumerate(basis):
        if average_support:
            lifted = 0.5 * (np.kron(op, identity) + np.kron(identity, op))
        else:
            lifted = np.kron(op, identity)
        ascended = isometry.conj().T @ lifted @ isometry
        for row, target in enumerate(basis):
            superoperator[row, col] = np.trace(target.conj().T @ ascended) / 2.0
    return superoperator, names


def narg_ascending_scaling_dimensions(
    *,
    j: float = 1.0,
    field: float = 1.0,
    block_factor: float = 2.0,
) -> IsingNARGAscending:
    """Estimate operator dimensions from a repeated two-site NARG layer."""
    isometry, energies = two_site_ising_isometry(j=j, field=field)
    superoperator, names = narg_ascending_superoperator(isometry)
    eigenvalues = np.linalg.eigvals(superoperator)
    with np.errstate(divide="ignore"):
        dimensions = -np.log(np.abs(eigenvalues)) / np.log(float(block_factor))
    order = np.argsort(dimensions)
    return IsingNARGAscending(
        isometry=isometry,
        block_energies=energies,
        superoperator=superoperator,
        eigenvalues=eigenvalues[order],
        dimensions=dimensions[order],
        operator_basis=names,
    )


__all__ = [
    "IsingFiniteSizeScaling",
    "IsingNARGAscending",
    "IsingNARGFixedLayerScaling",
    "TransverseFieldIsingNARG",
    "TransverseFieldIsingNARGResult",
    "TransverseFieldIsingNARGStep",
    "conditioned_spin_flip_operator",
    "finite_size_scaling_dimensions",
    "narg_fixed_layer_scaling_dimensions",
    "narg_fixed_layer_superoperator",
    "narg_ascending_scaling_dimensions",
    "narg_ascending_superoperator",
    "operator_symmetry_basis",
    "pauli_matrices",
    "transverse_field_ising_hamiltonian",
    "two_site_ising_isometry",
]
