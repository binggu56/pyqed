"""Matrix-free generalized eigensolvers for LETTA local updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class DavidsonDiagnostics:
    """Convergence information for a generalized Davidson solve."""

    converged: bool
    message: str
    iterations: int
    hamiltonian_matvecs: int
    metric_matvecs: int
    restarts: int
    residual_norm: float
    metric_norm: float
    projected_rank: int
    subspace_dimension: int
    energy_history: tuple[float, ...]
    residual_history: tuple[float, ...]


@dataclass(frozen=True)
class BlockDavidsonDiagnostics:
    """Convergence and action accounting for a recycled block solve."""

    converged: bool
    message: str
    iterations: int
    hamiltonian_action_calls: int
    hamiltonian_vector_products: int
    batch_action_calls: int
    scalar_action_calls: int
    restarts: int
    deterministic_augmentations: int
    residual_norm: float
    relative_residual: float
    subspace_dimension: int
    recycle_dimension: int
    energy_history: tuple[float, ...]
    residual_history: tuple[float, ...]

    @property
    def hamiltonian_matvecs(self):
        """Number of vectors acted on, including batched columns."""
        return self.hamiltonian_vector_products


def _as_finite_vector(value, size, *, name):
    vector = np.asarray(value)
    if vector.shape != (size,):
        raise ValueError(f"{name} must return a vector with shape {(size,)}.")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} returned a nonfinite vector.")
    return vector


def _as_finite_matrix(value, shape, *, name):
    matrix = np.asarray(value)
    if matrix.shape != shape:
        raise ValueError(f"{name} must return an array with shape {shape}.")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} returned a nonfinite array.")
    return matrix


def _orthogonalize(vector, basis, *, tolerance):
    """Twice apply Euclidean modified Gram--Schmidt."""
    dtype = np.result_type(
        np.asarray(vector).dtype,
        *[basis_vector.dtype for basis_vector in basis],
        np.float64,
    )
    vector = np.array(vector, dtype=dtype, copy=True)
    for _pass in range(2):
        for basis_vector in basis:
            vector -= basis_vector * np.vdot(basis_vector, vector)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= tolerance:
        return None
    return vector / norm


def _canonicalize_columns(vectors):
    vectors = np.array(vectors, copy=True)
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        value = vectors[pivot, column]
        if value != 0:
            vectors[:, column] *= np.conj(value) / abs(value)
    return vectors


def _prepare_preconditioner_blocks(blocks, size):
    prepared = []
    occupied = np.zeros(size, dtype=bool)
    for entry in () if blocks is None else blocks:
        if len(entry) != 2:
            raise ValueError(
                "each preconditioner block must be an (indices, matrix) pair."
            )
        indices, matrix = entry
        if isinstance(indices, slice):
            indices = np.arange(size)[indices]
        else:
            indices = np.asarray(indices, dtype=int)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("preconditioner block indices must be nonempty and 1D.")
        if np.any(indices < 0) or np.any(indices >= size):
            raise ValueError("preconditioner block index is out of range.")
        if np.unique(indices).size != indices.size or np.any(occupied[indices]):
            raise ValueError("preconditioner blocks must contain disjoint indices.")
        matrix = np.asarray(matrix)
        if matrix.shape != (indices.size, indices.size):
            raise ValueError(
                "a preconditioner block matrix must match its index count."
            )
        if np.any(~np.isfinite(matrix)):
            raise ValueError("preconditioner block contains nonfinite values.")
        occupied[indices] = True
        matrix = 0.5 * (matrix + matrix.T.conj())
        values, vectors = linalg.eigh(matrix, check_finite=False)
        prepared.append((indices, values, vectors))
    return prepared


def lowest_recycled_block_davidson(
    hamiltonian_action,
    initial_vectors,
    *,
    hamiltonian_batch_action=None,
    diagonal=None,
    preconditioner_blocks=None,
    block_size: int = 4,
    recycle_dimension: int = 4,
    tol: float = 1.0e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    max_subspace: int = 32,
    random_seed: int | None = 0,
):
    r"""Find the lowest eigenpair of a Hermitian operator by block Davidson.

    Trial and recycled vectors are columns of ``initial_vectors``.  New
    vectors are submitted together to ``hamiltonian_batch_action`` when that
    callback is supplied; otherwise ``hamiltonian_action`` is called once per
    column.  The returned recycled space contains the lowest Ritz vectors and
    can be passed directly as ``initial_vectors`` to the next solve.

    ``diagonal`` and ``preconditioner_blocks`` are optional Jacobi data.
    A block is an ``(indices, matrix)`` pair, where ``indices`` is a slice or a
    one-dimensional integer array.  Blocks must be disjoint and override the
    diagonal preconditioner on their support.

    The initial space is augmented by a deterministic DCT-probe block, which
    prevents immediate false convergence inside an exact invariant subspace.
    Correction breakdowns use the same complete deterministic probe sequence,
    offset by ``random_seed``, so a missed direction is eventually reached
    without stochastic behavior.

    Returns
    -------
    energy, eigenvector, recycle_vectors, diagnostics
        ``eigenvector`` and the columns of ``recycle_vectors`` have unit
        Euclidean norm.  Final energy and residual diagnostics use a fresh
        operator action.
    """
    initial_vectors = np.asarray(initial_vectors)
    if initial_vectors.ndim == 1:
        initial_vectors = initial_vectors[:, None]
    if (
        initial_vectors.ndim != 2
        or initial_vectors.shape[0] == 0
        or initial_vectors.shape[1] == 0
    ):
        raise ValueError(
            "initial_vectors must have shape (size, count) with nonzero dimensions."
        )
    if np.any(~np.isfinite(initial_vectors)):
        raise ValueError("initial_vectors must contain only finite values.")
    size = initial_vectors.shape[0]
    tol = float(tol)
    atol = float(atol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative.")
    block_size = min(size, int(block_size))
    recycle_dimension = min(size, int(recycle_dimension))
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if recycle_dimension < 1:
        raise ValueError("recycle_dimension must be positive.")
    maxiter = max(50, 4 * size) if maxiter is None else int(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    max_subspace = min(size, int(max_subspace))
    if max_subspace < min(size, 2):
        raise ValueError("max_subspace must be at least two for a nontrivial problem.")

    if diagonal is not None:
        diagonal = np.asarray(diagonal)
        if diagonal.shape != (size,):
            raise ValueError(f"diagonal must have shape {(size,)}.")
        if np.any(~np.isfinite(diagonal)):
            raise ValueError("diagonal must contain only finite values.")
    preconditioner_blocks = _prepare_preconditioner_blocks(
        preconditioner_blocks,
        size,
    )
    probe_cursor = 0 if random_seed is None else int(random_seed) % size

    action_calls = 0
    action_vectors = 0
    batch_calls = 0
    scalar_calls = 0

    def apply(vectors):
        nonlocal action_calls, action_vectors, batch_calls, scalar_calls
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors[:, None]
        count = vectors.shape[1]
        if hamiltonian_batch_action is not None:
            result = _as_finite_matrix(
                hamiltonian_batch_action(vectors),
                vectors.shape,
                name="hamiltonian_batch_action",
            )
            batch_calls += 1
            action_calls += 1
        else:
            result = np.column_stack(
                [
                    _as_finite_vector(
                        hamiltonian_action(vectors[:, column]),
                        size,
                        name="hamiltonian_action",
                    )
                    for column in range(count)
                ]
            )
            scalar_calls += count
            action_calls += count
        action_vectors += count
        return result

    breakdown_tolerance = 128.0 * np.finfo(float).eps * np.sqrt(size)

    def orthonormalize(candidates, basis, *, limit=None):
        candidates = np.asarray(candidates)
        if candidates.ndim == 1:
            candidates = candidates[:, None]
        accepted = []
        against = [basis[:, column] for column in range(basis.shape[1])]
        for column in range(candidates.shape[1]):
            vector = _orthogonalize(
                candidates[:, column],
                against + accepted,
                tolerance=breakdown_tolerance,
            )
            if vector is not None:
                accepted.append(vector)
            if limit is not None and len(accepted) >= limit:
                break
        if not accepted:
            return np.empty((size, 0), dtype=candidates.dtype)
        return np.column_stack(accepted)

    empty_basis = np.empty((size, 0), dtype=initial_vectors.dtype)
    initial_limit = max_subspace - min(block_size, max(0, max_subspace - 1))
    basis = orthonormalize(
        initial_vectors,
        empty_basis,
        limit=initial_limit,
    )
    if basis.shape[1] == 0:
        raise ValueError("initial_vectors are numerically zero or dependent.")

    augmentations = 0
    restarts = 0

    def deterministic_candidates(count):
        nonlocal probe_cursor
        grid = np.arange(size, dtype=float) + 0.5
        candidates = []
        while len(candidates) < count:
            frequency = probe_cursor % size
            probe_cursor += 1
            probe = np.cos(np.pi * grid * frequency / size)
            if np.iscomplexobj(basis):
                probe = probe.astype(np.result_type(basis.dtype, complex))
            candidates.append(probe)
        if not candidates:
            return np.empty((size, 0), dtype=basis.dtype)
        return np.column_stack(candidates)

    guard = orthonormalize(
        deterministic_candidates(size),
        basis,
        limit=min(block_size, max_subspace - basis.shape[1]),
    )
    if guard.shape[1]:
        basis = np.column_stack((basis, guard))
        augmentations += guard.shape[1]
    hamiltonian_basis = apply(basis)

    def append(candidates, *, deterministic=False, limit=None):
        nonlocal basis, hamiltonian_basis, augmentations
        capacity = max_subspace - basis.shape[1]
        if capacity <= 0:
            return 0
        if limit is not None:
            capacity = min(capacity, int(limit))
        candidates = orthonormalize(candidates, basis, limit=capacity)
        if candidates.shape[1] == 0:
            return 0
        actions = apply(candidates)
        basis = np.column_stack((basis, candidates))
        hamiltonian_basis = np.column_stack((hamiltonian_basis, actions))
        if deterministic:
            augmentations += candidates.shape[1]
        return candidates.shape[1]

    def precondition(residual, energy):
        correction = np.array(residual, copy=True)
        if diagonal is not None:
            denominator = energy - diagonal
            scale = np.maximum.reduce(
                (
                    np.abs(diagonal),
                    np.full(size, abs(energy)),
                    np.ones(size),
                )
            )
            safe = np.abs(denominator) > np.sqrt(np.finfo(float).eps) * scale
            correction[safe] = residual[safe] / denominator[safe]
            correction[~safe] = residual[~safe]
        for indices, values, vectors in preconditioner_blocks:
            denominator = energy - values
            scale = max(
                float(np.max(np.abs(values), initial=0.0)),
                abs(float(energy)),
                1.0,
            )
            cutoff = np.sqrt(np.finfo(float).eps) * scale
            inverse = np.zeros_like(denominator)
            safe = np.abs(denominator) > cutoff
            inverse[safe] = 1.0 / denominator[safe]
            correction[indices] = (
                vectors
                @ (inverse * (vectors.T.conj() @ residual[indices]))
            )
        return correction

    energy_history = []
    residual_history = []
    current_energy = np.nan
    current_vector = None
    current_residual = None
    current_scale = np.finfo(float).tiny
    projected_vectors = None
    projected_values = None
    converged = False
    message = "maximum iterations reached"
    iterations = 0

    for iteration in range(1, maxiter + 1):
        iterations = iteration
        projected = basis.T.conj() @ hamiltonian_basis
        projected = 0.5 * (projected + projected.T.conj())
        projected_values, coefficients = linalg.eigh(projected, check_finite=False)
        root_count = min(block_size, basis.shape[1])
        projected_vectors = basis @ coefficients[:, :root_count]
        projected_actions = hamiltonian_basis @ coefficients[:, :root_count]
        residuals = (
            projected_actions
            - projected_vectors * projected_values[None, :root_count]
        )
        residual_norms = np.linalg.norm(residuals, axis=0)
        scales = np.maximum.reduce(
            (
                np.linalg.norm(projected_actions, axis=0),
                np.abs(projected_values[:root_count]),
                np.full(root_count, np.finfo(float).tiny),
            )
        )
        current_energy = float(np.real(projected_values[0]))
        current_vector = projected_vectors[:, 0]
        current_residual = residuals[:, 0]
        current_scale = float(scales[0])
        residual_norm = float(residual_norms[0])
        energy_history.append(current_energy)
        residual_history.append(residual_norm)
        root_converged = residual_norm <= atol + tol * current_scale

        if root_converged:
            converged = True
            message = "converged"
            break
        if iteration == maxiter:
            break

        corrections = []
        for root in range(root_count):
            if residual_norms[root] <= atol + tol * scales[root]:
                continue
            corrections.append(
                precondition(residuals[:, root], float(projected_values[root]))
            )
        correction_block = (
            np.column_stack(corrections)
            if corrections
            else np.empty((size, 0), dtype=basis.dtype)
        )

        needed = max(1, min(block_size, max_subspace - recycle_dimension))
        if (
            basis.shape[1] + correction_block.shape[1] > max_subspace
            or basis.shape[1] == max_subspace
        ):
            retain = min(
                recycle_dimension,
                coefficients.shape[1],
                max(1, max_subspace - needed),
            )
            basis = basis @ coefficients[:, :retain]
            hamiltonian_basis = hamiltonian_basis @ coefficients[:, :retain]
            restarts += 1

        added = append(correction_block)
        if added:
            continue
        raw_residuals = residuals[
            :,
            residual_norms > atol + tol * scales,
        ]
        added = append(raw_residuals)
        if added:
            continue
        probes = deterministic_candidates(size)
        if append(probes, deterministic=True, limit=block_size):
            continue
        message = "Davidson search space exhausted before convergence"
        break

    if current_vector is None:
        raise ValueError(message)

    current_vector = current_vector / np.linalg.norm(current_vector)
    current_vector = _canonicalize_columns(current_vector[:, None])[:, 0]
    final_action = apply(current_vector[:, None])[:, 0]
    current_energy = float(np.real(np.vdot(current_vector, final_action)))
    current_residual = final_action - current_energy * current_vector
    residual_norm = float(np.linalg.norm(current_residual))
    current_scale = max(
        float(np.linalg.norm(final_action)),
        abs(current_energy),
        np.finfo(float).tiny,
    )
    relative_residual = residual_norm / current_scale
    converged = residual_norm <= atol + tol * current_scale
    if converged:
        message = "converged"
    elif message == "converged":
        message = "fresh final residual exceeds tolerance"

    projected = basis.T.conj() @ hamiltonian_basis
    projected = 0.5 * (projected + projected.T.conj())
    _, coefficients = linalg.eigh(projected, check_finite=False)
    retained = min(recycle_dimension, basis.shape[1])
    recycle_vectors = _canonicalize_columns(basis @ coefficients[:, :retained])
    diagnostics = BlockDavidsonDiagnostics(
        converged=bool(converged),
        message=message,
        iterations=iterations,
        hamiltonian_action_calls=action_calls,
        hamiltonian_vector_products=action_vectors,
        batch_action_calls=batch_calls,
        scalar_action_calls=scalar_calls,
        restarts=restarts,
        deterministic_augmentations=augmentations,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        subspace_dimension=basis.shape[1],
        recycle_dimension=recycle_vectors.shape[1],
        energy_history=tuple(energy_history),
        residual_history=tuple(residual_history),
    )
    return current_energy, current_vector, recycle_vectors, diagnostics


def _projected_lowest(hamiltonian, metric, *, metric_tol):
    """Solve a small Hermitian pencil on the positive range of its metric."""
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T.conj())
    metric = 0.5 * (metric + metric.T.conj())
    metric_values, metric_vectors = linalg.eigh(metric, check_finite=False)
    scale = max(float(np.linalg.norm(metric, ord=np.inf)), np.finfo(float).tiny)
    keep = metric_values > metric_tol * scale
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise ValueError("projected overlap metric is numerically singular.")
    metric_basis = metric_vectors[:, keep] / np.sqrt(metric_values[keep])[None, :]
    reduced_hamiltonian = metric_basis.T.conj() @ hamiltonian @ metric_basis
    reduced_hamiltonian = 0.5 * (reduced_hamiltonian + reduced_hamiltonian.T.conj())
    eigenvalues, eigenvectors = linalg.eigh(
        reduced_hamiltonian,
        subset_by_index=[0, 0],
        check_finite=False,
    )
    coefficients = metric_basis @ eigenvectors[:, 0]
    return float(np.real(eigenvalues[0])), coefficients, rank


def lowest_generalized_davidson(
    hamiltonian_action,
    metric_action,
    initial_vector,
    *,
    tol: float = 1.0e-10,
    atol: float = 0.0,
    metric_tol: float = 1.0e-12,
    maxiter: int | None = None,
    max_subspace: int = 32,
    random_seed: int | None = 0,
):
    r"""Find the lowest finite eigenpair of ``H x = E N x`` by actions.

    ``N`` may be positive semidefinite.  Every Davidson iteration solves the
    small projected pencil only on the positive range of its projected
    metric.  Neither full operator is assembled, and failure to converge is
    reported in :class:`DavidsonDiagnostics` without a dense fallback.

    The finite generalized problem is assumed to be well defined: in
    particular, null directions of ``N`` should also be null directions of
    ``H``.  This is the structure of LETTA's ``P_k^\dagger H P_k`` and
    ``P_k^\dagger P_k`` local operators.

    Parameters
    ----------
    hamiltonian_action, metric_action
        Callables accepting and returning one-dimensional vectors.
    initial_vector
        Initial trial vector.  If it lies entirely in ``ker(N)``, deterministic
        random probes are used to find a finite-metric starting direction.
    tol, atol
        Relative and absolute tolerances for ``||H x - E N x||``.
    metric_tol
        Relative cutoff for positive eigenvalues of each projected metric.
    maxiter
        Maximum number of projected solves.  Defaults to ``max(50, 4*n)``.
    max_subspace
        Maximum Davidson basis size before a two-vector restart.
    random_seed
        Seed used only for metric-null initialization or subspace breakdown.

    Returns
    -------
    energy, eigenvector, diagnostics
        The vector is normalized to ``x^\dagger N x = 1``.  Its reported
        energy and residual are recomputed from fresh final operator actions.
    """
    initial_vector = np.asarray(initial_vector)
    if initial_vector.ndim != 1 or initial_vector.size == 0:
        raise ValueError("initial_vector must be a nonempty one-dimensional array.")
    if np.any(~np.isfinite(initial_vector)):
        raise ValueError("initial_vector must contain only finite values.")
    size = initial_vector.size
    tol = float(tol)
    atol = float(atol)
    metric_tol = float(metric_tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative.")
    if not np.isfinite(metric_tol) or metric_tol < 0.0:
        raise ValueError("metric_tol must be finite and nonnegative.")
    maxiter = max(50, 4 * size) if maxiter is None else int(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    max_subspace = min(size, int(max_subspace))
    if max_subspace < min(size, 2):
        raise ValueError("max_subspace must be at least two for a nontrivial problem.")

    hamiltonian_matvecs = 0
    metric_matvecs = 0

    def apply_h(vector):
        nonlocal hamiltonian_matvecs
        hamiltonian_matvecs += 1
        return _as_finite_vector(
            hamiltonian_action(vector),
            size,
            name="hamiltonian_action",
        )

    def apply_n(vector):
        nonlocal metric_matvecs
        metric_matvecs += 1
        return _as_finite_vector(
            metric_action(vector),
            size,
            name="metric_action",
        )

    rng = np.random.default_rng(random_seed)
    basis = []
    hamiltonian_basis = []
    metric_basis = []
    breakdown_tolerance = 128.0 * np.finfo(float).eps * np.sqrt(size)

    def append_basis(vector):
        vector = _orthogonalize(
            vector,
            basis,
            tolerance=breakdown_tolerance,
        )
        if vector is None:
            return False
        h_vector = apply_h(vector)
        n_vector = apply_n(vector)
        basis.append(vector)
        hamiltonian_basis.append(h_vector)
        metric_basis.append(n_vector)
        return True

    def random_probe():
        dtype = np.result_type(
            initial_vector.dtype,
            *[vector.dtype for vector in hamiltonian_basis],
            *[vector.dtype for vector in metric_basis],
        )
        probe = rng.normal(size=size)
        if np.issubdtype(dtype, np.complexfloating):
            probe = probe + 1.0j * rng.normal(size=size)
        return probe

    if not append_basis(initial_vector):
        raise ValueError("initial_vector is numerically zero.")

    energy_history = []
    residual_history = []
    restarts = 0
    converged = False
    message = "maximum iterations reached"
    current_vector = None
    current_energy = np.nan
    current_residual = None
    current_metric_norm = 0.0
    current_rank = 0
    iterations = 0

    for iteration in range(1, maxiter + 1):
        iterations = iteration
        vectors = np.column_stack(basis)
        h_vectors = np.column_stack(hamiltonian_basis)
        n_vectors = np.column_stack(metric_basis)
        projected_h = vectors.T.conj() @ h_vectors
        projected_n = vectors.T.conj() @ n_vectors
        try:
            _projected_energy, coefficients, current_rank = _projected_lowest(
                projected_h,
                projected_n,
                metric_tol=metric_tol,
            )
        except ValueError:
            if len(basis) >= max_subspace:
                basis = []
                hamiltonian_basis = []
                metric_basis = []
                restarts += 1
            appended = False
            for _probe in range(max(8, min(size, 32))):
                if append_basis(random_probe()):
                    appended = True
                    break
            if not appended:
                message = "could not find a positive-metric trial direction"
                break
            continue

        current_vector = vectors @ coefficients
        h_current = h_vectors @ coefficients
        n_current = n_vectors @ coefficients
        metric_value = float(np.real(np.vdot(current_vector, n_current)))
        if not np.isfinite(metric_value) or metric_value <= 0.0:
            message = "Ritz vector has nonpositive metric norm"
            break
        normalization = np.sqrt(metric_value)
        current_vector /= normalization
        h_current /= normalization
        n_current /= normalization
        current_metric_norm = float(np.real(np.vdot(current_vector, n_current)))
        current_energy = float(
            np.real(np.vdot(current_vector, h_current)) / current_metric_norm
        )
        current_residual = h_current - current_energy * n_current
        residual_norm = float(np.linalg.norm(current_residual))
        residual_scale = max(
            float(np.linalg.norm(h_current)),
            abs(current_energy) * float(np.linalg.norm(n_current)),
            np.finfo(float).tiny,
        )
        energy_history.append(current_energy)
        residual_history.append(residual_norm)
        if residual_norm <= atol + tol * residual_scale:
            converged = True
            message = "converged"
            break
        if iteration == maxiter:
            break

        correction = _orthogonalize(
            current_residual,
            basis,
            tolerance=breakdown_tolerance,
        )
        if correction is None:
            for _probe in range(max(8, min(size, 32))):
                correction = _orthogonalize(
                    random_probe(),
                    basis,
                    tolerance=breakdown_tolerance,
                )
                if correction is not None:
                    break
        if correction is None:
            message = "Davidson subspace exhausted before convergence"
            break

        if len(basis) >= max_subspace:
            retained_vector = current_vector / np.linalg.norm(current_vector)
            retained_h = h_current / np.linalg.norm(current_vector)
            retained_n = n_current / np.linalg.norm(current_vector)
            basis = [retained_vector]
            hamiltonian_basis = [retained_h]
            metric_basis = [retained_n]
            restarts += 1
            correction = _orthogonalize(
                current_residual,
                basis,
                tolerance=breakdown_tolerance,
            )
            if correction is None:
                correction = _orthogonalize(
                    random_probe(),
                    basis,
                    tolerance=breakdown_tolerance,
                )
        if correction is None or not append_basis(correction):
            message = "Davidson restart failed to produce a new direction"
            break

    if current_vector is None:
        raise ValueError(message)

    h_final = apply_h(current_vector)
    n_final = apply_n(current_vector)
    final_metric = float(np.real(np.vdot(current_vector, n_final)))
    if not np.isfinite(final_metric) or final_metric <= 0.0:
        raise ValueError("final Davidson vector has nonpositive metric norm.")
    current_vector = current_vector / np.sqrt(final_metric)
    h_final = h_final / np.sqrt(final_metric)
    n_final = n_final / np.sqrt(final_metric)
    current_metric_norm = float(np.real(np.vdot(current_vector, n_final)))
    current_energy = float(
        np.real(np.vdot(current_vector, h_final)) / current_metric_norm
    )
    current_residual = h_final - current_energy * n_final
    residual_norm = float(np.linalg.norm(current_residual))
    residual_scale = max(
        float(np.linalg.norm(h_final)),
        abs(current_energy) * float(np.linalg.norm(n_final)),
        np.finfo(float).tiny,
    )
    converged = residual_norm <= atol + tol * residual_scale
    if converged:
        message = "converged"
    elif message == "converged":
        message = "fresh final residual exceeds tolerance"

    diagnostics = DavidsonDiagnostics(
        converged=bool(converged),
        message=message,
        iterations=iterations,
        hamiltonian_matvecs=hamiltonian_matvecs,
        metric_matvecs=metric_matvecs,
        restarts=restarts,
        residual_norm=residual_norm,
        metric_norm=current_metric_norm,
        projected_rank=current_rank,
        subspace_dimension=len(basis),
        energy_history=tuple(energy_history),
        residual_history=tuple(residual_history),
    )
    return current_energy, current_vector, diagnostics


__all__ = [
    "BlockDavidsonDiagnostics",
    "DavidsonDiagnostics",
    "lowest_generalized_davidson",
    "lowest_recycled_block_davidson",
]
