"""Bose-Hubbard NARG benchmark for a beam-splitter disentangler.

The main path applies a number-conserving beam-splitter before each recursive
block-branch NARG truncation.  A smaller block-site benchmark is kept as a
diagnostic reference.

    U(theta) = exp[theta (b_boundary^dag b_new - b_boundary b_new^dag)].

The reference is exact diagonalization in a fixed total-particle sector.  This
is intentionally small and dense: it tests whether a constrained disentangler
improves the basis chosen by the NARG truncation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

from pyqed.narg import rebranched_conditional_basis, rolling_conditional_basis_matrix
from pyqed.narg.bose_hubbard import (
    bose_hubbard_hamiltonian,
    boson_annihilation,
    fixed_number_basis,
)


@dataclass
class BoseHubbardDisentanglerResult:
    nsites: int
    block_sites: int
    theta_grid: np.ndarray
    energy_rms: np.ndarray
    subspace_fidelity: np.ndarray
    projected_energies: np.ndarray
    exact_energies: np.ndarray
    bare_theta: float
    best_scan_theta: float
    optimized_theta: float
    bare_rms: float
    best_scan_rms: float
    optimized_rms: float
    bare_fidelity: float
    best_scan_fidelity: float
    optimized_fidelity: float
    bare_energies: np.ndarray
    best_scan_energies: np.ndarray
    optimized_energies: np.ndarray
    kept_labels: list[tuple[int, int, int, float]]


@dataclass
class BoseHubbardStepwiseNARGResult:
    nsites: int
    nbosons: int
    keep: int
    theta_mode: str
    theta_grid: np.ndarray
    energy_rms: np.ndarray
    subspace_fidelity: np.ndarray
    projected_dims: np.ndarray
    exact_energies: np.ndarray
    bare_theta: float
    best_scan_theta: float
    optimized_theta: float
    bare_rms: float
    best_scan_rms: float
    optimized_rms: float
    bare_fidelity: float
    best_scan_fidelity: float
    optimized_fidelity: float
    bare_dim: int
    best_scan_dim: int
    optimized_dim: int
    best_scan_thetas: np.ndarray
    optimized_thetas: np.ndarray
    bare_energies: np.ndarray
    best_scan_energies: np.ndarray
    optimized_energies: np.ndarray


@dataclass
class StepwisePretruncationBasis:
    """Dense reference data for NARG after pre-truncation disentangling."""

    windows: list[np.ndarray]
    conditional_vectors: list[np.ndarray]
    basis: np.ndarray


@dataclass
class StepwiseGrowthNARGStep:
    """One block-branch NARG growth step."""

    site: int
    theta: float
    product_dim: int
    kept: int
    branch_kept: np.ndarray


@dataclass
class StepwiseGrowthNARGBasis:
    """Recursive NARG state after pre-truncation disentangling."""

    basis: np.ndarray
    qn: np.ndarray
    hamiltonian: np.ndarray
    boundary_b: np.ndarray
    steps: list[StepwiseGrowthNARGStep]


def _embed_local(op: np.ndarray, site: int, nsites: int, local_dim: int) -> np.ndarray:
    eye = np.eye(local_dim, dtype=complex)
    out = np.array([[1.0]], dtype=complex)
    for current in range(nsites):
        out = np.kron(out, op if current == site else eye)
    return out


def dense_bose_hubbard(
    nsites: int,
    nmax: int,
    *,
    t: float = 1.0,
    U: float = 1.0,
    mu: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the dense open-chain Bose-Hubbard Hamiltonian and site ``b`` ops."""
    local_dim = int(nmax) + 1
    nsites = int(nsites)
    b = boson_annihilation(local_dim).astype(complex)
    n = np.arange(local_dim, dtype=float)
    hloc = np.diag(0.5 * float(U) * n * (n - 1.0) - float(mu) * n).astype(complex)
    b_ops = [_embed_local(b, site, nsites, local_dim) for site in range(nsites)]
    hamiltonian = np.zeros((local_dim**nsites, local_dim**nsites), dtype=complex)
    for site in range(nsites):
        hamiltonian += _embed_local(hloc, site, nsites, local_dim)
    for site in range(nsites - 1):
        hamiltonian -= float(t) * (
            b_ops[site].conj().T @ b_ops[site + 1]
            + b_ops[site] @ b_ops[site + 1].conj().T
        )
    return 0.5 * (hamiltonian + hamiltonian.conj().T), b_ops


def fixed_number_indices(nsites: int, nbosons: int, nmax: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    basis = fixed_number_basis(nsites, nbosons, nmax)
    dims = (int(nmax) + 1,) * int(nsites)
    indices = np.asarray([np.ravel_multi_index(state, dims) for state in basis], dtype=int)
    return indices, basis


def exact_fixed_sector(
    hamiltonian: np.ndarray,
    nsites: int,
    nbosons: int,
    nmax: int,
    nroots: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    indices, basis = fixed_number_indices(nsites, nbosons, nmax)
    sector = hamiltonian[np.ix_(indices, indices)]
    values, vectors = np.linalg.eigh(0.5 * (sector + sector.conj().T))
    nroots = min(int(nroots), len(values))
    full_vectors = np.zeros((hamiltonian.shape[0], nroots), dtype=complex)
    for root in range(nroots):
        full_vectors[indices, root] = vectors[:, root]
    return values[:nroots].real, full_vectors, basis


def lowest_block_states(
    *,
    block_sites: int,
    nbosons: int,
    nmax: int,
    Dblock: int,
    t: float,
    U: float,
    mu: float,
) -> list[tuple[float, int, np.ndarray]]:
    """Return low block eigenstates across allowed block particle sectors."""
    local_dim = int(nmax) + 1
    h_block, _ = dense_bose_hubbard(block_sites, nmax, t=t, U=U, mu=mu)
    states: list[tuple[float, int, np.ndarray]] = []
    for qn in range(int(nbosons) + 1):
        basis = fixed_number_basis(block_sites, qn, nmax)
        if not basis:
            continue
        dims = (local_dim,) * int(block_sites)
        indices = np.asarray([np.ravel_multi_index(state, dims) for state in basis], dtype=int)
        sector = h_block[np.ix_(indices, indices)]
        values, vectors = np.linalg.eigh(0.5 * (sector + sector.conj().T))
        for root, value in enumerate(values):
            vector = np.zeros(local_dim**block_sites, dtype=complex)
            vector[indices] = vectors[:, root]
            states.append((float(np.real(value)), int(qn), vector))
    states.sort(key=lambda item: item[0])
    return states[: int(Dblock)]


def block_site_product_subspace(
    *,
    block_states: list[tuple[float, int, np.ndarray]],
    nbosons: int,
    nmax: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, float]]]:
    """Build columns |block state> tensor |new-site n> with total N fixed."""
    local_dim = int(nmax) + 1
    columns = []
    labels = []
    for state_index, (energy, block_number, block_vector) in enumerate(block_states):
        site_number = int(nbosons) - int(block_number)
        if 0 <= site_number < local_dim:
            site_vector = np.zeros(local_dim, dtype=complex)
            site_vector[site_number] = 1.0
            columns.append(np.kron(block_vector, site_vector))
            labels.append((int(state_index), int(block_number), int(site_number), float(energy)))
    if not columns:
        raise ValueError("No block-site product states satisfy the requested total boson number.")
    return np.column_stack(columns), labels


def two_site_beam_splitter(local_dim: int, theta: float) -> np.ndarray:
    """Return exp[theta (b0^dag b1 - b0 b1^dag)] on two local boson sites."""
    b = boson_annihilation(int(local_dim)).astype(complex)
    generator = np.kron(b.conj().T, b) - np.kron(b, b.conj().T)
    generator = 0.5 * (generator - generator.conj().T)
    return la.expm(float(theta) * generator)


def renormalized_beam_splitter(
    boundary_b: np.ndarray,
    local_b: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Return exp[theta (bbar^dag b - bbar b^dag)] on block x site."""
    boundary_b = np.asarray(boundary_b, dtype=complex)
    local_b = np.asarray(local_b, dtype=complex)
    generator = np.kron(boundary_b.conj().T, local_b) - np.kron(
        boundary_b,
        local_b.conj().T,
    )
    generator = 0.5 * (generator - generator.conj().T)
    return la.expm(float(theta) * generator)


def apply_two_site_unitary(
    vectors: np.ndarray,
    unitary: np.ndarray,
    *,
    nsites: int,
    local_dim: int,
    site_a: int,
    site_b: int,
) -> np.ndarray:
    """Apply a local two-site unitary to columns of product-basis vectors."""
    vectors = np.asarray(vectors, dtype=complex)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    nvec = vectors.shape[1]
    tensor = vectors.reshape((int(local_dim),) * int(nsites) + (nvec,))
    moved = np.moveaxis(tensor, (int(site_a), int(site_b)), (0, 1))
    moved_shape = moved.shape
    updated = unitary @ moved.reshape(int(local_dim) ** 2, -1)
    updated = updated.reshape(moved_shape)
    restored = np.moveaxis(updated, (0, 1), (int(site_a), int(site_b)))
    return restored.reshape(vectors.shape)


def two_site_beam_splitter(local_dim: int, theta: float) -> np.ndarray:
    """Return exp[theta (b0^dag b1 - b0 b1^dag)] on two local boson sites."""
    b = boson_annihilation(int(local_dim)).astype(complex)
    generator = np.kron(b.conj().T, b) - np.kron(b, b.conj().T)
    generator = 0.5 * (generator - generator.conj().T)
    return la.expm(float(theta) * generator)


def apply_two_site_unitary(
    vectors: np.ndarray,
    unitary: np.ndarray,
    *,
    nsites: int,
    local_dim: int,
    site_a: int,
    site_b: int,
) -> np.ndarray:
    """Apply a local two-site unitary to columns of product-basis vectors."""
    vectors = np.asarray(vectors, dtype=complex)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    nvec = vectors.shape[1]
    tensor = vectors.reshape((int(local_dim),) * int(nsites) + (nvec,))
    moved = np.moveaxis(tensor, (int(site_a), int(site_b)), (0, 1))
    moved_shape = moved.shape
    updated = unitary @ moved.reshape(int(local_dim) ** 2, -1)
    updated = updated.reshape(moved_shape)
    restored = np.moveaxis(updated, (0, 1), (int(site_a), int(site_b)))
    return restored.reshape(vectors.shape)


def transform_operator_by_two_site_unitary(
    operator: np.ndarray,
    unitary: np.ndarray,
    *,
    nsites: int,
    local_dim: int,
    site_a: int,
    site_b: int,
) -> np.ndarray:
    """Return ``U^dag operator U`` for a local two-site unitary."""
    right = apply_two_site_unitary(
        operator.conj().T,
        unitary.conj().T,
        nsites=nsites,
        local_dim=local_dim,
        site_a=site_a,
        site_b=site_b,
    ).conj().T
    transformed = apply_two_site_unitary(
        right,
        unitary.conj().T,
        nsites=nsites,
        local_dim=local_dim,
        site_a=site_a,
        site_b=site_b,
    )
    return 0.5 * (transformed + transformed.conj().T)


def stepwise_transformed_hamiltonian(
    hamiltonian: np.ndarray,
    thetas: np.ndarray,
    *,
    nsites: int,
    local_dim: int,
) -> np.ndarray:
    """Apply a beam-splitter disentangler on every growth bond."""
    transformed = np.asarray(hamiltonian, dtype=complex)
    for bond, theta in enumerate(np.asarray(thetas, dtype=float)):
        unitary = two_site_beam_splitter(local_dim, float(theta))
        transformed = transform_operator_by_two_site_unitary(
            transformed,
            unitary,
            nsites=nsites,
            local_dim=local_dim,
            site_a=bond,
            site_b=bond + 1,
        )
    return transformed


def _theta_vector(theta: float | np.ndarray, nsites: int) -> np.ndarray:
    """Return one disentangler angle for each nearest-neighbor growth bond."""
    values = np.asarray(theta, dtype=float)
    if values.ndim == 0:
        return np.full(int(nsites) - 1, float(values), dtype=float)
    values = values.reshape(-1)
    if values.shape != (int(nsites) - 1,):
        raise ValueError("theta must be a scalar or have length nsites - 1.")
    return values


def stepwise_pretruncated_window_hamiltonians(
    *,
    nsites: int,
    nmax: int,
    t: float,
    U: float,
    mu: float,
    thetas: np.ndarray,
) -> list[np.ndarray]:
    """Build each local window after its disentanglers and before truncation."""
    local_dim = int(nmax) + 1
    h3, _ = dense_bose_hubbard(3, nmax, t=t, U=U, mu=mu)
    windows = []
    for start in range(int(nsites) - 2):
        hwin = h3.copy()
        for local_bond, theta in enumerate((thetas[start], thetas[start + 1])):
            hwin = transform_operator_by_two_site_unitary(
                hwin,
                two_site_beam_splitter(local_dim, float(theta)),
                nsites=3,
                local_dim=local_dim,
                site_a=local_bond,
                site_b=local_bond + 1,
            )
        windows.append(hwin)
    return windows


def stepwise_pretruncation_basis(
    *,
    nsites: int,
    nmax: int,
    t: float,
    U: float,
    mu: float,
    keep: int,
    theta: float | np.ndarray,
) -> StepwisePretruncationBasis:
    """Apply the disentangler before each conditional NARG truncation.

    For active site ``i``, the local three-site problem is first transformed by
    the beam splitters on ``(i, i + 1)`` and ``(i + 1, i + 2)``.  Only after that
    local similarity transformation do we solve the branch eigenproblems and
    keep ``D=keep`` conditional states.
    """
    nsites = int(nsites)
    nmax = int(nmax)
    local_dim = nmax + 1
    dims = (local_dim,) * nsites
    thetas = _theta_vector(theta, nsites)
    windows = stepwise_pretruncated_window_hamiltonians(
        nsites=nsites,
        nmax=nmax,
        t=t,
        U=U,
        mu=mu,
        thetas=thetas,
    )
    conditional_vectors = []
    for window in windows:
        _energies, vectors = rebranched_conditional_basis(
            window,
            (local_dim, local_dim, local_dim),
            keep,
        )
        conditional_vectors.append(vectors)
    basis = rolling_conditional_basis_matrix(conditional_vectors, dims, mode="rebranched")
    return StepwisePretruncationBasis(
        windows=windows,
        conditional_vectors=conditional_vectors,
        basis=basis,
    )


def stepwise_window_hamiltonians(
    *,
    nsites: int,
    nmax: int,
    t: float,
    U: float,
    mu: float,
    thetas: np.ndarray,
) -> list[np.ndarray]:
    """Backward-compatible alias for pre-truncation transformed windows."""
    return stepwise_pretruncated_window_hamiltonians(
        nsites=nsites,
        nmax=nmax,
        t=t,
        U=U,
        mu=mu,
        thetas=thetas,
    )


def _orthonormal_columns(matrix: np.ndarray, tol: float = 1.0e-10) -> np.ndarray:
    """Return an orthonormal basis for the column span."""
    if matrix.size == 0:
        return np.zeros_like(matrix)
    u, s, _vh = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0:
        return u[:, :0]
    keep = s > max(float(tol), float(tol) * s[0])
    return u[:, keep]


def _allowed_counts_for_target(
    *,
    built_sites: int,
    total_sites: int,
    nbosons: int,
    nmax: int,
) -> set[int]:
    remaining = int(total_sites) - int(built_sites)
    low = max(0, int(nbosons) - remaining * int(nmax))
    high = min(int(nbosons), int(built_sites) * int(nmax))
    return set(range(low, high + 1))


def _conditional_branch_truncation(
    h_product: np.ndarray,
    qn_product: np.ndarray,
    branch_numbers: np.ndarray,
    *,
    local_dim: int,
    keep: int,
    allowed_numbers: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep low block states conditionally for each new-site occupation."""
    h_product = 0.5 * (h_product + h_product.conj().T)
    qn_product = np.asarray(qn_product, dtype=int)
    branch_numbers = np.asarray(branch_numbers, dtype=int)
    allowed_numbers = set(int(number) for number in allowed_numbers)
    columns = []
    qn_keep = []
    branch_kept = np.zeros(int(local_dim), dtype=int)
    for branch in range(int(local_dim)):
        branch_rows = np.flatnonzero(
            (branch_numbers == branch)
            & np.isin(qn_product, np.fromiter(allowed_numbers, dtype=int))
        )
        if branch_rows.size == 0:
            continue
        roots = []
        for number in sorted(set(qn_product[branch_rows].tolist()) & allowed_numbers):
            rows = branch_rows[qn_product[branch_rows] == number]
            block = h_product[np.ix_(rows, rows)]
            values, vectors = np.linalg.eigh(0.5 * (block + block.conj().T))
            for root, value in enumerate(values):
                vector = np.zeros(h_product.shape[0], dtype=complex)
                vector[rows] = vectors[:, root]
                roots.append((float(np.real(value)), int(number), vector))
        roots.sort(key=lambda item: item[0])
        for _value, number, vector in roots[: min(int(keep), len(roots))]:
            columns.append(vector)
            qn_keep.append(number)
            branch_kept[branch] += 1
    if not columns:
        raise ValueError("No conditional NARG states remain after branch truncation.")
    return np.column_stack(columns), np.asarray(qn_keep, dtype=int), branch_kept


def stepwise_growth_pretruncation_basis(
    *,
    nsites: int,
    nbosons: int,
    nmax: int,
    t: float,
    U: float,
    mu: float,
    keep: int,
    theta: float | np.ndarray,
) -> StepwiseGrowthNARGBasis:
    """Grow a block-branch NARG basis with renormalized disentanglers.

    At growth step ``site``, the bond disentangler is built from the current
    renormalized boundary operator ``bbar`` and the bare operator on the new
    site.  The transformed product Hamiltonian is truncated branch-by-branch,
    then ``H`` and ``bbar`` are both renormalized into the kept NARG basis.
    """
    nsites = int(nsites)
    nmax = int(nmax)
    local_dim = nmax + 1
    keep = int(keep)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    if keep < 1:
        raise ValueError("keep must be positive.")

    thetas = _theta_vector(theta, nsites)
    b = boson_annihilation(local_dim).astype(complex)
    nloc = np.arange(local_dim, dtype=int)
    n = nloc.astype(float)
    hloc = np.diag(0.5 * float(U) * n * (n - 1.0) - float(mu) * n).astype(complex)

    initial_allowed = _allowed_counts_for_target(
        built_sites=1,
        total_sites=nsites,
        nbosons=nbosons,
        nmax=nmax,
    )
    initial_rows = np.asarray(
        [idx for idx, number in enumerate(nloc) if int(number) in initial_allowed],
        dtype=int,
    )
    if initial_rows.size == 0:
        raise ValueError("No initial Bose-Hubbard NARG states satisfy the target number.")

    basis = np.eye(local_dim, dtype=complex)[:, initial_rows]
    qn = nloc[initial_rows].copy()
    h_block = hloc[np.ix_(initial_rows, initial_rows)].copy()
    boundary_b = b[np.ix_(initial_rows, initial_rows)].copy()
    steps: list[StepwiseGrowthNARGStep] = []

    for site in range(1, nsites):
        block_dim = h_block.shape[0]
        eye_local = np.eye(local_dim, dtype=complex)
        eye_block = np.eye(block_dim, dtype=complex)
        product_basis = np.kron(basis, eye_local)
        h_product = np.kron(h_block, eye_local) + np.kron(eye_block, hloc)
        h_product -= float(t) * (
            np.kron(boundary_b.conj().T, b)
            + np.kron(boundary_b, b.conj().T)
        )
        unitary = renormalized_beam_splitter(boundary_b, b, float(thetas[site - 1]))
        transformed_product = unitary.conj().T @ h_product @ unitary
        transformed_product = 0.5 * (transformed_product + transformed_product.conj().T)
        qn_product = np.repeat(qn, local_dim) + np.tile(nloc, qn.size)
        branch_numbers = np.tile(nloc, qn.size)
        allowed = _allowed_counts_for_target(
            built_sites=site + 1,
            total_sites=nsites,
            nbosons=nbosons,
            nmax=nmax,
        )
        conditional_basis, qn, branch_kept = _conditional_branch_truncation(
            transformed_product,
            qn_product,
            branch_numbers,
            local_dim=local_dim,
            keep=keep,
            allowed_numbers=allowed,
        )
        basis = product_basis @ (unitary @ conditional_basis)
        h_block = conditional_basis.conj().T @ transformed_product @ conditional_basis
        h_block = 0.5 * (h_block + h_block.conj().T)
        new_boundary = unitary.conj().T @ np.kron(eye_block, b) @ unitary
        boundary_b = conditional_basis.conj().T @ new_boundary @ conditional_basis
        steps.append(
            StepwiseGrowthNARGStep(
                site=site,
                theta=float(thetas[site - 1]),
                product_dim=int(h_product.shape[0]),
                kept=int(basis.shape[1]),
                branch_kept=branch_kept,
            )
        )

    return StepwiseGrowthNARGBasis(
        basis=basis,
        qn=qn,
        hamiltonian=h_block,
        boundary_b=boundary_b,
        steps=steps,
    )


def projected_stepwise_narg_observables(
    *,
    nsites: int,
    nbosons: int,
    nmax: int,
    t: float,
    U: float,
    mu: float,
    keep: int,
    theta: float | np.ndarray,
    exact_energies: np.ndarray,
    exact_vectors: np.ndarray,
    nroots: int,
) -> tuple[float, float, np.ndarray, int]:
    """Evaluate recursive NARG with renormalized pre-truncation disentanglers."""
    pretruncation = stepwise_growth_pretruncation_basis(
        nsites=nsites,
        nbosons=nbosons,
        nmax=nmax,
        t=t,
        U=U,
        mu=mu,
        keep=keep,
        theta=theta,
    )
    fixed_indices, _ = fixed_number_indices(nsites, nbosons, nmax)
    target_columns = np.flatnonzero(pretruncation.qn == int(nbosons))
    if target_columns.size == 0:
        raise ValueError("Recursive NARG produced no states in the target number sector.")
    projected = pretruncation.hamiltonian[np.ix_(target_columns, target_columns)]
    projected = 0.5 * (projected + projected.conj().T)
    basis_n = _orthonormal_columns(pretruncation.basis[fixed_indices, :][:, target_columns])
    if basis_n.shape[1] == 0:
        raise ValueError("Fixed-number projection of the NARG basis is empty.")
    values = np.linalg.eigvalsh(projected)[: min(int(nroots), projected.shape[0])].real
    reference = exact_energies[: len(values)].real
    rms = float(np.sqrt(np.mean((values - reference) ** 2)))
    fidelity = float(
        np.linalg.norm(exact_vectors[:, : len(values)].conj().T @ basis_n, ord="fro") ** 2
        / len(values)
    )
    return rms, fidelity, values, int(basis_n.shape[1])


def _projected_observables(
    *,
    hamiltonian: np.ndarray,
    projector: np.ndarray,
    exact_energies: np.ndarray,
    exact_vectors: np.ndarray,
    theta: float,
    nsites: int,
    local_dim: int,
    site_a: int,
    site_b: int,
    nroots: int,
) -> tuple[float, float, np.ndarray]:
    unitary = two_site_beam_splitter(local_dim, theta)
    basis = apply_two_site_unitary(
        projector,
        unitary,
        nsites=nsites,
        local_dim=local_dim,
        site_a=site_a,
        site_b=site_b,
    )
    projected = basis.conj().T @ hamiltonian @ basis
    projected = 0.5 * (projected + projected.conj().T)
    values = np.linalg.eigvalsh(projected)[: min(int(nroots), projected.shape[0])].real
    reference = exact_energies[: len(values)]
    rms = float(np.sqrt(np.mean((values - reference) ** 2)))
    fidelity = float(
        np.linalg.norm(exact_vectors[:, : len(values)].conj().T @ basis, ord="fro") ** 2
        / len(values)
    )
    return rms, fidelity, values


def run_block_site_benchmark(
    *,
    nsites: int = 3,
    nbosons: int = 3,
    nmax: int = 3,
    t: float = 1.0,
    U: float = 1.0,
    mu: float = 0.0,
    Dblock: int = 6,
    nroots: int = 3,
    theta_min: float = -0.75,
    theta_max: float = 0.75,
    theta_points: int = 121,
    optimize: bool = True,
    optimizer_xatol: float = 1.0e-6,
) -> BoseHubbardDisentanglerResult:
    """Compare bare and beam-splitter block-site subspaces for an L-site chain."""
    nsites = int(nsites)
    if nsites < 3:
        raise ValueError("nsites must be at least 3 for the block-site benchmark.")
    block_sites = nsites - 1
    local_dim = int(nmax) + 1
    hamiltonian, b_ops = dense_bose_hubbard(nsites, nmax, t=t, U=U, mu=mu)
    exact_energies, exact_vectors, _ = exact_fixed_sector(
        hamiltonian,
        nsites,
        nbosons,
        nmax,
        nroots,
    )
    block_states = lowest_block_states(
        block_sites=block_sites,
        nbosons=nbosons,
        nmax=nmax,
        Dblock=Dblock,
        t=t,
        U=U,
        mu=mu,
    )
    projector, labels = block_site_product_subspace(
        block_states=block_states,
        nbosons=nbosons,
        nmax=nmax,
    )
    if projector.shape[1] < min(nroots, len(exact_energies)):
        raise ValueError("Projected subspace has fewer states than requested roots.")

    boundary = block_sites - 1
    new_site = block_sites

    theta_grid = np.linspace(float(theta_min), float(theta_max), int(theta_points))
    energy_rms = np.empty_like(theta_grid)
    fidelity = np.empty_like(theta_grid)
    projected_energies = np.empty((len(theta_grid), int(nroots)), dtype=float)
    for idx, theta in enumerate(theta_grid):
        rms, fid, values = _projected_observables(
            hamiltonian=hamiltonian,
            projector=projector,
            exact_energies=exact_energies,
            exact_vectors=exact_vectors,
            theta=float(theta),
            nsites=nsites,
            local_dim=local_dim,
            site_a=boundary,
            site_b=new_site,
            nroots=nroots,
        )
        energy_rms[idx] = rms
        fidelity[idx] = fid
        projected_energies[idx, : len(values)] = values

    bare_index = int(np.argmin(np.abs(theta_grid)))
    best_scan_index = int(np.argmin(energy_rms))
    optimized_theta = float(theta_grid[best_scan_index])
    best_scan_thetas = np.full(nsites - 1, optimized_theta, dtype=float)
    optimized_thetas = best_scan_thetas.copy()
    optimized_rms = float(energy_rms[best_scan_index])
    optimized_fidelity = float(fidelity[best_scan_index])
    optimized_energies = projected_energies[best_scan_index].copy()

    if optimize:
        def objective(theta: float) -> float:
            rms, _, _ = _projected_observables(
                hamiltonian=hamiltonian,
                projector=projector,
                exact_energies=exact_energies,
                exact_vectors=exact_vectors,
                theta=float(theta),
                nsites=nsites,
                local_dim=local_dim,
                site_a=boundary,
                site_b=new_site,
                nroots=nroots,
            )
            return rms

        result = opt.minimize_scalar(
            objective,
            bounds=(float(theta_grid[0]), float(theta_grid[-1])),
            method="bounded",
            options={"xatol": float(optimizer_xatol)},
        )
        if result.success:
            optimized_theta = float(result.x)
            optimized_rms, optimized_fidelity, optimized_energies = _projected_observables(
                hamiltonian=hamiltonian,
                projector=projector,
                exact_energies=exact_energies,
                exact_vectors=exact_vectors,
                theta=optimized_theta,
                nsites=nsites,
                local_dim=local_dim,
                site_a=boundary,
                site_b=new_site,
                nroots=nroots,
            )

    return BoseHubbardDisentanglerResult(
        nsites=nsites,
        block_sites=block_sites,
        theta_grid=theta_grid,
        energy_rms=energy_rms,
        subspace_fidelity=fidelity,
        projected_energies=projected_energies,
        exact_energies=exact_energies,
        bare_theta=float(theta_grid[bare_index]),
        best_scan_theta=float(theta_grid[best_scan_index]),
        optimized_theta=optimized_theta,
        bare_rms=float(energy_rms[bare_index]),
        best_scan_rms=float(energy_rms[best_scan_index]),
        optimized_rms=float(optimized_rms),
        bare_fidelity=float(fidelity[bare_index]),
        best_scan_fidelity=float(fidelity[best_scan_index]),
        optimized_fidelity=float(optimized_fidelity),
        bare_energies=projected_energies[bare_index].copy(),
        best_scan_energies=projected_energies[best_scan_index].copy(),
        optimized_energies=np.asarray(optimized_energies, dtype=float),
        kept_labels=labels,
    )


def run_stepwise_narg_benchmark(
    *,
    nsites: int = 4,
    nbosons: int = 4,
    nmax: int = 3,
    t: float = 1.0,
    U: float = 1.0,
    mu: float = 0.0,
    keep: int = 2,
    nroots: int = 3,
    theta_min: float = -0.75,
    theta_max: float = 0.75,
    theta_points: int = 81,
    theta_mode: str = "uniform",
    optimize: bool = True,
    optimizer_xatol: float = 1.0e-5,
    optimizer_maxiter: int = 100,
) -> BoseHubbardStepwiseNARGResult:
    """Run recursive block-branch NARG with a disentangler at every bond."""
    nsites = int(nsites)
    nbosons = int(nbosons)
    nmax = int(nmax)
    theta_mode = str(theta_mode).lower().replace("_", "-")
    if theta_mode not in {"uniform", "per-step"}:
        raise ValueError("theta_mode must be 'uniform' or 'per-step'.")
    sector_h_sparse, _ = bose_hubbard_hamiltonian(
        nsites,
        nbosons,
        t=t,
        U=U,
        nmax=nmax,
        mu=mu,
    )
    sector_h = sector_h_sparse.toarray().astype(complex)
    exact_values, exact_vectors = np.linalg.eigh(0.5 * (sector_h + sector_h.conj().T))
    nroots = min(int(nroots), len(exact_values))
    exact_values = exact_values[:nroots].real
    exact_vectors = exact_vectors[:, :nroots]

    theta_grid = np.linspace(float(theta_min), float(theta_max), int(theta_points))
    energy_rms = np.empty_like(theta_grid)
    fidelity = np.empty_like(theta_grid)
    projected_dims = np.empty(theta_grid.shape, dtype=int)
    projected_energies = np.full((theta_grid.size, nroots), np.nan, dtype=float)
    for idx, theta in enumerate(theta_grid):
        rms, fid, values, dim = projected_stepwise_narg_observables(
            nsites=nsites,
            nbosons=nbosons,
            nmax=nmax,
            t=t,
            U=U,
            mu=mu,
            keep=keep,
            theta=float(theta),
            exact_energies=exact_values,
            exact_vectors=exact_vectors,
            nroots=nroots,
        )
        energy_rms[idx] = rms
        fidelity[idx] = fid
        projected_dims[idx] = dim
        projected_energies[idx, : len(values)] = values

    bare_index = int(np.argmin(np.abs(theta_grid)))
    best_scan_index = int(np.argmin(energy_rms))
    optimized_theta = float(theta_grid[best_scan_index])
    best_scan_thetas = np.full(nsites - 1, optimized_theta, dtype=float)
    optimized_thetas = best_scan_thetas.copy()
    optimized_rms = float(energy_rms[best_scan_index])
    optimized_fidelity = float(fidelity[best_scan_index])
    optimized_dim = int(projected_dims[best_scan_index])
    optimized_energies = projected_energies[best_scan_index].copy()

    if optimize:
        def uniform_objective(theta: float) -> float:
            rms, _fid, _values, _dim = projected_stepwise_narg_observables(
                nsites=nsites,
                nbosons=nbosons,
                nmax=nmax,
                t=t,
                U=U,
                mu=mu,
                keep=keep,
                theta=float(theta),
                exact_energies=exact_values,
                exact_vectors=exact_vectors,
                nroots=nroots,
            )
            return rms

        result = opt.minimize_scalar(
            uniform_objective,
            bounds=(float(theta_grid[0]), float(theta_grid[-1])),
            method="bounded",
            options={"xatol": float(optimizer_xatol)},
        )
        if result.success:
            candidate_theta = float(result.x)
            candidate_rms, candidate_fidelity, candidate_energies, candidate_dim = (
                projected_stepwise_narg_observables(
                    nsites=nsites,
                    nbosons=nbosons,
                    nmax=nmax,
                    t=t,
                    U=U,
                    mu=mu,
                    keep=keep,
                    theta=candidate_theta,
                    exact_energies=exact_values,
                    exact_vectors=exact_vectors,
                    nroots=nroots,
                )
            )
            if candidate_rms <= optimized_rms:
                optimized_theta = candidate_theta
                optimized_thetas = np.full(nsites - 1, optimized_theta, dtype=float)
                optimized_rms = candidate_rms
                optimized_fidelity = candidate_fidelity
                optimized_energies = candidate_energies
                optimized_dim = candidate_dim

        if theta_mode == "per-step":
            def vector_objective(thetas: np.ndarray) -> float:
                clipped = np.clip(np.asarray(thetas, dtype=float), theta_grid[0], theta_grid[-1])
                rms, _fid, _values, _dim = projected_stepwise_narg_observables(
                    nsites=nsites,
                    nbosons=nbosons,
                    nmax=nmax,
                    t=t,
                    U=U,
                    mu=mu,
                    keep=keep,
                    theta=clipped,
                    exact_energies=exact_values,
                    exact_vectors=exact_vectors,
                    nroots=nroots,
                )
                return rms

            result = opt.minimize(
                vector_objective,
                optimized_thetas,
                method="Powell",
                bounds=[(float(theta_grid[0]), float(theta_grid[-1]))] * (nsites - 1),
                options={
                    "xtol": float(optimizer_xatol),
                    "ftol": float(optimizer_xatol),
                    "maxiter": int(optimizer_maxiter),
                    "disp": False,
                },
            )
            if result.success or np.isfinite(result.fun):
                candidate_thetas = np.clip(
                    np.asarray(result.x, dtype=float),
                    theta_grid[0],
                    theta_grid[-1],
                )
                candidate_rms, candidate_fidelity, candidate_energies, candidate_dim = (
                    projected_stepwise_narg_observables(
                        nsites=nsites,
                        nbosons=nbosons,
                        nmax=nmax,
                        t=t,
                        U=U,
                        mu=mu,
                        keep=keep,
                        theta=candidate_thetas,
                        exact_energies=exact_values,
                        exact_vectors=exact_vectors,
                        nroots=nroots,
                    )
                )
                if candidate_rms <= optimized_rms:
                    optimized_thetas = candidate_thetas
                    optimized_theta = float(np.mean(optimized_thetas))
                    optimized_rms = candidate_rms
                    optimized_fidelity = candidate_fidelity
                    optimized_energies = candidate_energies
                    optimized_dim = candidate_dim

    return BoseHubbardStepwiseNARGResult(
        nsites=nsites,
        nbosons=nbosons,
        keep=int(keep),
        theta_mode=theta_mode,
        theta_grid=theta_grid,
        energy_rms=energy_rms,
        subspace_fidelity=fidelity,
        projected_dims=projected_dims,
        exact_energies=exact_values,
        bare_theta=float(theta_grid[bare_index]),
        best_scan_theta=float(theta_grid[best_scan_index]),
        optimized_theta=optimized_theta,
        bare_rms=float(energy_rms[bare_index]),
        best_scan_rms=float(energy_rms[best_scan_index]),
        optimized_rms=float(optimized_rms),
        bare_fidelity=float(fidelity[bare_index]),
        best_scan_fidelity=float(fidelity[best_scan_index]),
        optimized_fidelity=float(optimized_fidelity),
        bare_dim=int(projected_dims[bare_index]),
        best_scan_dim=int(projected_dims[best_scan_index]),
        optimized_dim=int(optimized_dim),
        best_scan_thetas=best_scan_thetas,
        optimized_thetas=optimized_thetas,
        bare_energies=projected_energies[bare_index].copy(),
        best_scan_energies=projected_energies[best_scan_index].copy(),
        optimized_energies=np.asarray(optimized_energies, dtype=float),
    )


def save_outputs(
    result: BoseHubbardDisentanglerResult,
    output_dir: Path,
    prefix: str = "bose_hubbard_disentangler",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(result.kept_labels, dtype=float)
    np.savez(
        output_dir / f"{prefix}.npz",
        theta_grid=result.theta_grid,
        energy_rms=result.energy_rms,
        subspace_fidelity=result.subspace_fidelity,
        projected_energies=result.projected_energies,
        exact_energies=result.exact_energies,
        bare_theta=result.bare_theta,
        best_scan_theta=result.best_scan_theta,
        optimized_theta=result.optimized_theta,
        bare_rms=result.bare_rms,
        best_scan_rms=result.best_scan_rms,
        optimized_rms=result.optimized_rms,
        bare_fidelity=result.bare_fidelity,
        best_scan_fidelity=result.best_scan_fidelity,
        optimized_fidelity=result.optimized_fidelity,
        bare_energies=result.bare_energies,
        best_scan_energies=result.best_scan_energies,
        optimized_energies=result.optimized_energies,
        kept_labels=labels,
    )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(result.theta_grid, result.energy_rms)
    ax.axvline(0.0, color="k", linestyle=":", label="bare")
    ax.axvline(result.best_scan_theta, color="tab:green", linestyle="--", label="best scan")
    ax.axvline(result.optimized_theta, color="tab:red", linestyle="-.", label="Brent")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("RMS low-energy error")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_energy_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(result.theta_grid, result.subspace_fidelity)
    ax.axvline(0.0, color="k", linestyle=":", label="bare")
    ax.axvline(result.best_scan_theta, color="tab:green", linestyle="--", label="best scan")
    ax.axvline(result.optimized_theta, color="tab:red", linestyle="-.", label="Brent")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("low-energy subspace fidelity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_fidelity.png", dpi=180)
    plt.close(fig)


def save_stepwise_outputs(
    result: BoseHubbardStepwiseNARGResult,
    output_dir: Path,
    prefix: str = "bose_hubbard_stepwise_narg",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{prefix}.npz",
        nsites=result.nsites,
        nbosons=result.nbosons,
        keep=result.keep,
        theta_mode=result.theta_mode,
        theta_grid=result.theta_grid,
        energy_rms=result.energy_rms,
        subspace_fidelity=result.subspace_fidelity,
        projected_dims=result.projected_dims,
        exact_energies=result.exact_energies,
        bare_theta=result.bare_theta,
        best_scan_theta=result.best_scan_theta,
        optimized_theta=result.optimized_theta,
        bare_rms=result.bare_rms,
        best_scan_rms=result.best_scan_rms,
        optimized_rms=result.optimized_rms,
        bare_fidelity=result.bare_fidelity,
        best_scan_fidelity=result.best_scan_fidelity,
        optimized_fidelity=result.optimized_fidelity,
        bare_dim=result.bare_dim,
        best_scan_dim=result.best_scan_dim,
        optimized_dim=result.optimized_dim,
        best_scan_thetas=result.best_scan_thetas,
        optimized_thetas=result.optimized_thetas,
        bare_energies=result.bare_energies,
        best_scan_energies=result.best_scan_energies,
        optimized_energies=result.optimized_energies,
    )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(result.theta_grid, result.energy_rms)
    ax.axvline(0.0, color="k", linestyle=":", label="bare")
    ax.axvline(result.best_scan_theta, color="tab:green", linestyle="--", label="best scan")
    ax.axvline(result.optimized_theta, color="tab:red", linestyle="-.", label="Brent")
    ax.set_xlabel(r"uniform step $\theta$")
    ax.set_ylabel("fixed-N RMS low-energy error")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_energy_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(result.theta_grid, result.subspace_fidelity)
    ax.axvline(0.0, color="k", linestyle=":", label="bare")
    ax.axvline(result.best_scan_theta, color="tab:green", linestyle="--", label="best scan")
    ax.axvline(result.optimized_theta, color="tab:red", linestyle="-.", label="Brent")
    ax.set_xlabel(r"uniform step $\theta$")
    ax.set_ylabel("fixed-N low-energy subspace fidelity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_fidelity.png", dpi=180)
    plt.close(fig)


def format_summary(result: BoseHubbardDisentanglerResult) -> str:
    lines = [
        "Bose-Hubbard NARG-style beam-splitter disentangler benchmark",
        f"  kept block labels       = {result.kept_labels}",
        f"  bare theta              = {result.bare_theta:.8f}",
        f"  best scan theta         = {result.best_scan_theta:.8f}",
        f"  Brent theta             = {result.optimized_theta:.8f}",
        f"  RMS error bare          = {result.bare_rms:.8e}",
        f"  RMS error best scan     = {result.best_scan_rms:.8e}",
        f"  RMS error Brent         = {result.optimized_rms:.8e}",
        f"  fidelity bare           = {result.bare_fidelity:.8f}",
        f"  fidelity Brent          = {result.optimized_fidelity:.8f}",
        "",
        "  level      exact           bare            best-scan       Brent",
    ]
    for level, (exact, bare, scan, opt_energy) in enumerate(
        zip(
            result.exact_energies,
            result.bare_energies,
            result.best_scan_energies,
            result.optimized_energies,
        )
    ):
        lines.append(
            f"  {level:3d}  {exact:14.8f} {bare:14.8f} {scan:14.8f} {opt_energy:14.8f}"
        )
    return "\n".join(lines)


def format_stepwise_summary(result: BoseHubbardStepwiseNARGResult) -> str:
    optimized_thetas = np.array2string(
        result.optimized_thetas,
        precision=5,
        separator=", ",
        suppress_small=False,
    )
    lines = [
        "Bose-Hubbard stepwise block-branch NARG + beam-splitter disentangler",
        f"  L, N, keep             = {result.nsites}, {result.nbosons}, {result.keep}",
        f"  theta mode             = {result.theta_mode}",
        f"  bare theta             = {result.bare_theta:.8f}",
        f"  best scan theta        = {result.best_scan_theta:.8f}",
        f"  optimized theta mean   = {result.optimized_theta:.8f}",
        f"  optimized thetas       = {optimized_thetas}",
        f"  projected dim bare     = {result.bare_dim}",
        f"  projected dim optimized= {result.optimized_dim}",
        f"  RMS error bare         = {result.bare_rms:.8e}",
        f"  RMS error best scan    = {result.best_scan_rms:.8e}",
        f"  RMS error optimized    = {result.optimized_rms:.8e}",
        f"  fidelity bare          = {result.bare_fidelity:.8f}",
        f"  fidelity optimized     = {result.optimized_fidelity:.8f}",
        "",
        "  level      exact           bare            best-scan       Brent",
    ]
    for level, (exact, bare, scan, opt_energy) in enumerate(
        zip(
            result.exact_energies,
            result.bare_energies,
            result.best_scan_energies,
            result.optimized_energies,
        )
    ):
        lines.append(
            f"  {level:3d}  {exact:14.8f} {bare:14.8f} {scan:14.8f} {opt_energy:14.8f}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("stepwise-narg", "block-site"), default="stepwise-narg")
    parser.add_argument("--nsites", type=int, default=4)
    parser.add_argument("--nbosons", type=int, default=3)
    parser.add_argument("--nmax", type=int, default=3)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--onsite-u", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--dblock", type=int, default=6)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--nroots", type=int, default=3)
    parser.add_argument("--theta-min", type=float, default=-0.75)
    parser.add_argument("--theta-max", type=float, default=0.75)
    parser.add_argument("--theta-points", type=int, default=121)
    parser.add_argument("--theta-mode", choices=("uniform", "per-step"), default="uniform")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--optimizer-xatol", type=float, default=1.0e-6)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/bose_hubbard_disentangler"))
    parser.add_argument("--prefix", default="bose_hubbard_disentangler")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.algorithm == "block-site":
        result = run_block_site_benchmark(
            nsites=args.nsites,
            nbosons=args.nbosons,
            nmax=args.nmax,
            t=args.hopping,
            U=args.onsite_u,
            mu=args.mu,
            Dblock=args.dblock,
            nroots=args.nroots,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            theta_points=args.theta_points,
            theta_mode=args.theta_mode,
            optimize=not args.no_optimize,
            optimizer_xatol=args.optimizer_xatol,
            optimizer_maxiter=args.optimizer_maxiter,
        )
        save_outputs(result, args.output_dir, prefix=args.prefix)
        print(format_summary(result))
    else:
        result = run_stepwise_narg_benchmark(
            nsites=args.nsites,
            nbosons=args.nbosons,
            nmax=args.nmax,
            t=args.hopping,
            U=args.onsite_u,
            mu=args.mu,
            keep=args.keep,
            nroots=args.nroots,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            theta_points=args.theta_points,
            optimize=not args.no_optimize,
            optimizer_xatol=args.optimizer_xatol,
        )
        save_stepwise_outputs(result, args.output_dir, prefix=args.prefix)
        print(format_stepwise_summary(result))
    print(f"\nSaved outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
