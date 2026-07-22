"""Dense conditional two-site NARG reference algorithms.

The routines here implement the small-system version of a true two-site NARG
update: when the second branch site is introduced, the conditional basis of the
active mode is recomputed in the presence of that second site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar


@dataclass
class ConditionalTwoSiteResult:
    energies: np.ndarray
    vectors: np.ndarray
    projected_hamiltonian: np.ndarray
    basis: np.ndarray
    conditional_vectors: np.ndarray
    mode: str
    dressing: str = "none"
    undressed_energies: np.ndarray | None = None
    discarded_residual_norm: float = 0.0
    dressing_scale: float = 0.0
    dressing_mixing: float = 0.0


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


def _conditional_direct_sum_basis(conditional_vectors):
    vectors = np.asarray(conditional_vectors, dtype=complex)
    if vectors.ndim < 3:
        raise ValueError(
            "conditional_vectors must have shape (*branches, active_dim, states)."
        )
    branch_shape = vectors.shape[:-2]
    active_dim, states = vectors.shape[-2:]
    nbranches = int(np.prod(branch_shape))
    flat = vectors.reshape(nbranches, active_dim, states)
    basis = np.zeros((active_dim * nbranches, nbranches * states), dtype=complex)
    active = np.arange(active_dim, dtype=int)
    state = np.arange(states, dtype=int)
    for branch in range(nbranches):
        rows = active * nbranches + branch
        columns = branch * states + state
        basis[np.ix_(rows, columns)] = flat[branch]
    return basis


def conditional_cc_dress_basis(
    hamiltonian,
    conditional_vectors,
    keep,
    *,
    level_shift=0.0,
    max_scale=4.0,
):
    """Dress a conditional direct-sum basis with one state-specific CC response.

    The complete conditional eigenvectors define branchwise retained and
    discarded spaces.  A linear discarded-space response is computed for the
    projected ground state, then one retained direction per branch is rotated
    toward its discarded response.  A scalar line search keeps the update
    variational and preserves the number and direct-sum structure of the
    conditional basis.
    """
    vectors = np.asarray(conditional_vectors, dtype=complex)
    if vectors.ndim < 3 or vectors.shape[-2] != vectors.shape[-1]:
        raise ValueError(
            "conditional_vectors must contain complete branch bases with shape "
            "(*branches, active_dim, active_dim)."
        )
    branch_shape = vectors.shape[:-2]
    active_dim = int(vectors.shape[-2])
    nbranches = int(np.prod(branch_shape))
    expected_dim = active_dim * nbranches
    hamiltonian = _as_hermitian(hamiltonian)
    if hamiltonian.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"hamiltonian must have shape {(expected_dim, expected_dim)}, "
            f"got {hamiltonian.shape}."
        )
    keep = min(int(keep), active_dim)
    if keep < 1:
        raise ValueError("keep must be positive.")
    if float(max_scale) <= 0.0:
        raise ValueError("max_scale must be positive.")

    retained = vectors[..., :keep]
    discarded = vectors[..., keep:]
    basis = _conditional_direct_sum_basis(retained)
    projected = _as_hermitian(basis.conj().T @ (hamiltonian @ basis))
    values, coefficients = np.linalg.eigh(projected)
    undressed_energy = float(values[0])

    if keep == active_dim:
        return retained, {
            "undressed_energy": undressed_energy,
            "discarded_residual_norm": 0.0,
            "scale": 0.0,
        }

    complement = _conditional_direct_sum_basis(discarded)
    hpq = basis.conj().T @ (hamiltonian @ complement)
    hqq = _as_hermitian(complement.conj().T @ (hamiltonian @ complement))
    ground = coefficients[:, 0]
    residual = hpq.conj().T @ ground
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1.0e-14:
        return retained, {
            "undressed_energy": undressed_energy,
            "discarded_residual_norm": residual_norm,
            "scale": 0.0,
        }

    shifted = hqq + (float(level_shift) - undressed_energy) * np.eye(hqq.shape[0])
    response = np.linalg.lstsq(shifted, -residual, rcond=1.0e-12)[0]

    retained_flat = retained.reshape(nbranches, active_dim, keep)
    discarded_flat = discarded.reshape(nbranches, active_dim, active_dim - keep)
    ground_flat = ground.reshape(nbranches, keep)
    response_flat = response.reshape(nbranches, active_dim - keep)
    branch_data = []
    for branch in range(nbranches):
        retained_norm = float(np.linalg.norm(ground_flat[branch]))
        response_norm = float(np.linalg.norm(response_flat[branch]))
        if retained_norm <= 1.0e-14 or response_norm <= 1.0e-14:
            branch_data.append(None)
            continue
        retained_direction = ground_flat[branch] / retained_norm
        retained_complement = null_space(retained_direction.conj()[None, :])
        rotation = np.column_stack((retained_direction, retained_complement))
        rotated = retained_flat[branch] @ rotation
        discarded_direction = discarded_flat[branch] @ (
            response_flat[branch] / response_norm
        )
        branch_data.append(
            (rotated, discarded_direction, response_norm / retained_norm)
        )

    def dressed_vectors(scale):
        out = retained_flat.copy()
        for branch, data in enumerate(branch_data):
            if data is None:
                continue
            rotated, discarded_direction, amplitude_ratio = data
            angle = np.arctan(float(scale) * amplitude_ratio)
            out[branch, :, 0] = (
                np.cos(angle) * rotated[:, 0]
                + np.sin(angle) * discarded_direction
            )
            out[branch, :, 1:] = rotated[:, 1:]
        return out.reshape(*branch_shape, active_dim, keep)

    def ground_energy(scale):
        trial_basis = _conditional_direct_sum_basis(dressed_vectors(scale))
        trial_hamiltonian = _as_hermitian(
            trial_basis.conj().T @ (hamiltonian @ trial_basis)
        )
        return float(np.linalg.eigvalsh(trial_hamiltonian)[0])

    optimum = minimize_scalar(
        ground_energy,
        bounds=(0.0, float(max_scale)),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    if not optimum.success or float(optimum.fun) >= undressed_energy - 1.0e-13:
        scale = 0.0
        dressed = retained
    else:
        scale = float(optimum.x)
        dressed = dressed_vectors(scale)
    return dressed, {
        "undressed_energy": undressed_energy,
        "discarded_residual_norm": residual_norm,
        "scale": scale,
    }


def conditional_cc_transition_basis(
    hamiltonian,
    conditional_vectors,
    keep,
    *,
    level_shift=0.0,
):
    """Add one cross-branch conditional-CC response at fixed model dimension.

    Unlike ``conditional_cc_dress_basis``, this response may map a retained
    state on branch ``t`` into a discarded state on another branch ``s``.  The
    returned isometry is therefore a conditional basis followed by one
    transition layer, rather than a strict branch-direct-sum basis.
    """
    vectors = np.asarray(conditional_vectors, dtype=complex)
    if vectors.ndim < 3 or vectors.shape[-2] != vectors.shape[-1]:
        raise ValueError(
            "conditional_vectors must contain complete branch bases with shape "
            "(*branches, active_dim, active_dim)."
        )
    active_dim = int(vectors.shape[-2])
    keep = min(int(keep), active_dim)
    if keep < 1:
        raise ValueError("keep must be positive.")

    retained = vectors[..., :keep]
    basis = _conditional_direct_sum_basis(retained)
    hamiltonian = _as_hermitian(hamiltonian)
    if (
        hamiltonian.shape[0] != basis.shape[0]
        or hamiltonian.shape[1] != basis.shape[0]
    ):
        raise ValueError("hamiltonian shape is inconsistent with conditional_vectors.")
    projected = _as_hermitian(basis.conj().T @ (hamiltonian @ basis))
    values, coefficients = np.linalg.eigh(projected)
    undressed_energy = float(values[0])
    if keep == active_dim:
        return basis, {
            "undressed_energy": undressed_energy,
            "discarded_residual_norm": 0.0,
            "mixing": 0.0,
        }

    discarded = vectors[..., keep:]
    complement = _conditional_direct_sum_basis(discarded)
    hpq = basis.conj().T @ (hamiltonian @ complement)
    hqq = _as_hermitian(complement.conj().T @ (hamiltonian @ complement))
    ground = coefficients[:, 0]
    residual = hpq.conj().T @ ground
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1.0e-14:
        return basis, {
            "undressed_energy": undressed_energy,
            "discarded_residual_norm": residual_norm,
            "mixing": 0.0,
        }

    shifted = hqq + (float(level_shift) - undressed_energy) * np.eye(hqq.shape[0])
    response = np.linalg.lstsq(shifted, -residual, rcond=1.0e-12)[0]
    response_norm = float(np.linalg.norm(response))
    if response_norm <= 1.0e-14:
        return basis, {
            "undressed_energy": undressed_energy,
            "discarded_residual_norm": residual_norm,
            "mixing": 0.0,
        }
    response_direction = response / response_norm
    coupling = np.vdot(ground, hpq @ response_direction)
    response_energy = float(
        np.real(np.vdot(response_direction, hqq @ response_direction))
    )
    pair_hamiltonian = np.array(
        [
            [undressed_energy, coupling],
            [coupling.conjugate(), response_energy],
        ],
        dtype=complex,
    )
    _pair_values, pair_vectors = np.linalg.eigh(pair_hamiltonian)
    retained_weight, discarded_weight = pair_vectors[:, 0]
    dressed_ground = (
        retained_weight * (basis @ ground)
        + discarded_weight * (complement @ response_direction)
    )
    dressed_basis = np.column_stack((dressed_ground, basis @ coefficients[:, 1:]))
    return dressed_basis, {
        "undressed_energy": undressed_energy,
        "discarded_residual_norm": residual_norm,
        "mixing": float(abs(discarded_weight) ** 2),
    }


def conditional_two_site_narg(
    h01,
    h012,
    dims,
    keep,
    *,
    mode="rebranched",
    nroots=1,
    dressing=None,
    cc_level_shift=0.0,
    cc_max_scale=4.0,
):
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
    dressing
        ``"conditional_cc"`` applies an experimental cross-branch,
        state-specific rank-one CC response to a rebranched basis.
        ``"conditional_cc_branch"`` restricts the response to rotations
        within each branch.
    """
    dims = tuple(int(d) for d in dims)
    if len(dims) != 3 or any(d < 1 for d in dims):
        raise ValueError("dims must contain three positive local dimensions.")
    keep = int(keep)
    if keep < 1:
        raise ValueError("keep must be positive.")
    keep = min(keep, dims[0])
    mode = str(mode).lower().replace("-", "_")
    dressing_key = (
        "none" if dressing is None else str(dressing).lower().replace("-", "_")
    )
    if dressing_key in {"none", "off", "false"}:
        dressing_key = "none"
    elif dressing_key in {
        "conditional_cc",
        "conditional_cc_transition",
        "transition_cc",
        "cc",
        "rank1_cc",
    }:
        dressing_key = "conditional_cc"
    elif dressing_key in {"conditional_cc_branch", "branch_cc"}:
        dressing_key = "conditional_cc_branch"
    else:
        raise ValueError(
            "dressing must be None, 'conditional_cc', or "
            "'conditional_cc_branch'."
        )

    h012 = _as_hermitian(h012)
    undressed_values = None
    discarded_residual_norm = 0.0
    dressing_scale = 0.0
    dressing_mixing = 0.0

    if mode in {"sequential", "one_site", "old"}:
        if dressing_key != "none":
            raise ValueError(
                "conditional_cc dressing currently requires mode='rebranched'."
            )
        _energies, conditional_vectors = sequential_conditional_basis(h01, dims, keep)
        basis = _sequential_basis_matrix(conditional_vectors, dims)
        canonical_mode = "sequential"
    elif mode in {"rebranched", "two_site", "true_two_site"}:
        if dressing_key == "none":
            _energies, conditional_vectors = rebranched_conditional_basis(h012, dims, keep)
        else:
            _energies, complete_vectors = rebranched_conditional_basis(
                h012,
                dims,
                dims[0],
            )
            undressed_vectors = complete_vectors[..., :keep]
            undressed_basis = _rebranched_basis_matrix(undressed_vectors, dims)
            undressed_projected = _as_hermitian(
                undressed_basis.conj().T @ (h012 @ undressed_basis)
            )
            undressed_values, _ = _lowest_eigenvectors(undressed_projected, nroots)
            if dressing_key == "conditional_cc_branch":
                conditional_vectors, diagnostics = conditional_cc_dress_basis(
                    h012,
                    complete_vectors,
                    keep,
                    level_shift=cc_level_shift,
                    max_scale=cc_max_scale,
                )
                basis = _rebranched_basis_matrix(conditional_vectors, dims)
                dressing_scale = diagnostics["scale"]
            else:
                basis, diagnostics = conditional_cc_transition_basis(
                    h012,
                    complete_vectors,
                    keep,
                    level_shift=cc_level_shift,
                )
                # Cross-branch T^{st} cannot be encoded by branch-local
                # conditional vectors alone; ``basis`` carries that layer.
                conditional_vectors = undressed_vectors
                dressing_mixing = diagnostics["mixing"]
            discarded_residual_norm = diagnostics["discarded_residual_norm"]
        if dressing_key == "none":
            basis = _rebranched_basis_matrix(conditional_vectors, dims)
        canonical_mode = "rebranched"
    else:
        raise ValueError("mode must be 'sequential' or 'rebranched'.")

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
        dressing=dressing_key,
        undressed_energies=undressed_values,
        discarded_residual_norm=discarded_residual_norm,
        dressing_scale=dressing_scale,
        dressing_mixing=dressing_mixing,
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
    "conditional_cc_dress_basis",
    "conditional_cc_transition_basis",
    "conditional_two_site_narg",
    "rebranched_conditional_basis",
    "rolling_conditional_basis_matrix",
    "rolling_conditional_narg",
    "rolling_rebranched_conditional_basis",
    "rolling_sequential_conditional_basis",
    "sequential_conditional_basis",
]
