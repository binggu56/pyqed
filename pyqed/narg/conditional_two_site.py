"""Dense conditional two-site NARG reference algorithms.

The routines here implement the small-system version of a true two-site NARG
update: when the second branch site is introduced, the conditional basis of the
active mode is recomputed in the presence of that second site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConditionalTwoSiteResult:
    energies: np.ndarray
    vectors: np.ndarray
    projected_hamiltonian: np.ndarray
    basis: np.ndarray
    conditional_vectors: np.ndarray
    mode: str


@dataclass
class RollingConditionalNARGResult:
    energies: np.ndarray
    vectors: np.ndarray
    projected_hamiltonian: np.ndarray
    basis: np.ndarray
    conditional_vectors: list[np.ndarray]
    mode: str


def _as_hermitian(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    return 0.5 * (matrix + matrix.T.conj())


def _lowest_eigenvectors(matrix, keep):
    matrix = _as_hermitian(matrix)
    values, vectors = np.linalg.eigh(matrix)
    keep = min(int(keep), matrix.shape[0])
    return values[:keep], vectors[:, :keep]


def _branch_block_two_mode(hamiltonian, dims, branch):
    d0, d1 = (int(d) for d in dims)
    tensor = np.asarray(hamiltonian, dtype=complex).reshape(d0, d1, d0, d1)
    return tensor[:, branch, :, branch]


def _branch_block_three_mode(hamiltonian, dims, branch1, branch2):
    d0, d1, d2 = (int(d) for d in dims)
    tensor = np.asarray(hamiltonian, dtype=complex).reshape(d0, d1, d2, d0, d1, d2)
    return tensor[:, branch1, branch2, :, branch1, branch2]


def _basis_column_index(state, dims):
    d0, d1, d2 = (int(d) for d in dims)
    n0, n1, n2 = state
    return (int(n0) * d1 + int(n1)) * d2 + int(n2)


def sequential_conditional_basis(h01, dims, keep):
    """Conditional active-mode basis depending only on site 1."""
    d0, d1, _d2 = (int(d) for d in dims)
    keep = min(int(keep), d0)
    vectors = np.zeros((d1, d0, keep), dtype=complex)
    energies = np.zeros((d1, keep), dtype=float)
    for n1 in range(d1):
        energies[n1], vectors[n1] = _lowest_eigenvectors(
            _branch_block_two_mode(h01, (d0, d1), n1), keep
        )
    return energies, vectors


def rebranched_conditional_basis(h012, dims, keep):
    """Conditional active-mode basis depending on both sites 1 and 2."""
    d0, d1, d2 = (int(d) for d in dims)
    keep = min(int(keep), d0)
    vectors = np.zeros((d1, d2, d0, keep), dtype=complex)
    energies = np.zeros((d1, d2, keep), dtype=float)
    for n1 in range(d1):
        for n2 in range(d2):
            energies[n1, n2], vectors[n1, n2] = _lowest_eigenvectors(
                _branch_block_three_mode(h012, dims, n1, n2), keep
            )
    return energies, vectors


def _sequential_basis_matrix(conditional_vectors, dims):
    d0, d1, d2 = (int(d) for d in dims)
    keep = conditional_vectors.shape[-1]
    basis = np.zeros((d0 * d1 * d2, d1 * d2 * keep), dtype=complex)
    col = 0
    for n1 in range(d1):
        for n2 in range(d2):
            for alpha in range(keep):
                for n0 in range(d0):
                    row = _basis_column_index((n0, n1, n2), dims)
                    basis[row, col] = conditional_vectors[n1, n0, alpha]
                col += 1
    return basis


def _rebranched_basis_matrix(conditional_vectors, dims):
    d0, d1, d2 = (int(d) for d in dims)
    keep = conditional_vectors.shape[-1]
    basis = np.zeros((d0 * d1 * d2, d1 * d2 * keep), dtype=complex)
    col = 0
    for n1 in range(d1):
        for n2 in range(d2):
            for alpha in range(keep):
                for n0 in range(d0):
                    row = _basis_column_index((n0, n1, n2), dims)
                    basis[row, col] = conditional_vectors[n1, n2, n0, alpha]
                col += 1
    return basis


def conditional_two_site_narg(h01, h012, dims, keep, *, mode="rebranched", nroots=1):
    """Project a three-mode Hamiltonian into a conditional NARG basis.

    Parameters
    ----------
    h01
        Two-mode Hamiltonian for modes ``(0, 1)`` used by the ordinary
        sequential update to choose the mode-0 conditional basis.
    h012
        Full three-mode Hamiltonian for modes ``(0, 1, 2)``.
    dims
        Local dimensions ``(d0, d1, d2)``.
    keep
        Number of conditional mode-0 states kept per branch.
    mode
        ``"sequential"`` keeps the old site-0 basis conditioned only on site 1.
        ``"rebranched"`` recomputes that site-0 basis for each ``(site 1,
        site 2)`` branch.
    """
    dims = tuple(int(d) for d in dims)
    if len(dims) != 3 or any(d < 1 for d in dims):
        raise ValueError("dims must contain three positive local dimensions.")
    keep = int(keep)
    if keep < 1:
        raise ValueError("keep must be positive.")
    mode = str(mode).lower().replace("-", "_")

    if mode in {"sequential", "one_site", "old"}:
        _energies, conditional_vectors = sequential_conditional_basis(h01, dims, keep)
        basis = _sequential_basis_matrix(conditional_vectors, dims)
        canonical_mode = "sequential"
    elif mode in {"rebranched", "two_site", "true_two_site"}:
        _energies, conditional_vectors = rebranched_conditional_basis(h012, dims, keep)
        basis = _rebranched_basis_matrix(conditional_vectors, dims)
        canonical_mode = "rebranched"
    else:
        raise ValueError("mode must be 'sequential' or 'rebranched'.")

    h012 = _as_hermitian(h012)
    projected = basis.conj().T @ (h012 @ basis)
    projected = _as_hermitian(projected)
    values, vectors = _lowest_eigenvectors(projected, nroots)
    return ConditionalTwoSiteResult(
        energies=values,
        vectors=vectors,
        projected_hamiltonian=projected,
        basis=basis,
        conditional_vectors=conditional_vectors,
        mode=canonical_mode,
    )


def _mixed_radix_index(state, dims):
    index = 0
    for value, dim in zip(state, dims):
        index = index * int(dim) + int(value)
    return index


def rolling_sequential_conditional_basis(pair_hamiltonians, dims, keep):
    """Site-i conditional bases depending only on site i+1."""
    dims = tuple(int(d) for d in dims)
    if len(pair_hamiltonians) != max(0, len(dims) - 2):
        raise ValueError("pair_hamiltonians must have one entry for each rolling active site.")
    out = []
    for idx, h_pair in enumerate(pair_hamiltonians):
        d0 = dims[idx]
        d1 = dims[idx + 1]
        local_keep = min(int(keep), d0)
        vectors = np.zeros((d1, d0, local_keep), dtype=complex)
        for branch in range(d1):
            _energies, vectors[branch] = _lowest_eigenvectors(
                _branch_block_two_mode(h_pair, (d0, d1), branch), local_keep
            )
        out.append(vectors)
    return out


def rolling_rebranched_conditional_basis(triple_hamiltonians, dims, keep):
    """Site-i conditional bases depending on sites i+1 and i+2."""
    dims = tuple(int(d) for d in dims)
    if len(triple_hamiltonians) != max(0, len(dims) - 2):
        raise ValueError("triple_hamiltonians must have one entry for each rolling active site.")
    out = []
    for idx, h_triple in enumerate(triple_hamiltonians):
        d0 = dims[idx]
        d1 = dims[idx + 1]
        d2 = dims[idx + 2]
        _energies, vectors = rebranched_conditional_basis(h_triple, (d0, d1, d2), keep)
        out.append(vectors)
    return out


def rolling_conditional_basis_matrix(conditional_vectors, dims, *, mode="rebranched"):
    """Build the full triangular rolling conditional basis matrix."""
    dims = tuple(int(d) for d in dims)
    if len(dims) < 2:
        raise ValueError("rolling conditional NARG requires at least two modes.")
    active_count = len(dims) - 2
    if len(conditional_vectors) != active_count:
        raise ValueError("conditional_vectors must have len(dims) - 2 entries.")
    if active_count == 0:
        return np.eye(int(np.prod(dims)), dtype=complex)

    mode = str(mode).lower().replace("-", "_")
    keep_dims = [int(vec.shape[-1]) for vec in conditional_vectors]
    tail_dims = dims[-2:]
    nrows = int(np.prod(dims))
    ncols = int(np.prod(keep_dims) * np.prod(tail_dims))
    basis = np.zeros((nrows, ncols), dtype=complex)

    col = 0
    for alphas in np.ndindex(*keep_dims):
        for tail in np.ndindex(*tail_dims):
            for state_prefix in np.ndindex(*dims[:-2]):
                state = tuple(state_prefix) + tuple(tail)
                amp = 1.0 + 0.0j
                for site, alpha in enumerate(alphas):
                    if mode in {"sequential", "one_site", "old"}:
                        amp *= conditional_vectors[site][state[site + 1], state[site], alpha]
                    elif mode in {"rebranched", "two_site", "true_two_site"}:
                        amp *= conditional_vectors[site][
                            state[site + 1], state[site + 2], state[site], alpha
                        ]
                    else:
                        raise ValueError("mode must be 'sequential' or 'rebranched'.")
                    if amp == 0:
                        break
                if amp != 0:
                    basis[_mixed_radix_index(state, dims), col] = amp
            col += 1
    return basis


def rolling_conditional_narg(
    hamiltonian,
    dims,
    keep,
    *,
    pair_hamiltonians=None,
    triple_hamiltonians=None,
    mode="rebranched",
    nroots=1,
):
    """Project a chain Hamiltonian into a rolling conditional NARG basis."""
    dims = tuple(int(d) for d in dims)
    if len(dims) < 2 or any(d < 1 for d in dims):
        raise ValueError("dims must contain at least two positive local dimensions.")
    keep = int(keep)
    if keep < 1:
        raise ValueError("keep must be positive.")
    mode = str(mode).lower().replace("-", "_")

    if mode in {"sequential", "one_site", "old"}:
        if pair_hamiltonians is None:
            raise ValueError("pair_hamiltonians are required for sequential rolling mode.")
        conditional_vectors = rolling_sequential_conditional_basis(pair_hamiltonians, dims, keep)
        canonical_mode = "sequential"
    elif mode in {"rebranched", "two_site", "true_two_site"}:
        if triple_hamiltonians is None:
            raise ValueError("triple_hamiltonians are required for rebranched rolling mode.")
        conditional_vectors = rolling_rebranched_conditional_basis(triple_hamiltonians, dims, keep)
        canonical_mode = "rebranched"
    else:
        raise ValueError("mode must be 'sequential' or 'rebranched'.")

    basis = rolling_conditional_basis_matrix(conditional_vectors, dims, mode=canonical_mode)
    hamiltonian = _as_hermitian(hamiltonian)
    projected = basis.conj().T @ (hamiltonian @ basis)
    projected = _as_hermitian(projected)
    values, vectors = _lowest_eigenvectors(projected, nroots)
    return RollingConditionalNARGResult(
        energies=values,
        vectors=vectors,
        projected_hamiltonian=projected,
        basis=basis,
        conditional_vectors=conditional_vectors,
        mode=canonical_mode,
    )


__all__ = [
    "ConditionalTwoSiteResult",
    "RollingConditionalNARGResult",
    "conditional_two_site_narg",
    "rebranched_conditional_basis",
    "rolling_conditional_basis_matrix",
    "rolling_conditional_narg",
    "rolling_rebranched_conditional_basis",
    "rolling_sequential_conditional_basis",
    "sequential_conditional_basis",
]
