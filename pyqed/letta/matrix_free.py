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


def _as_finite_vector(value, size, *, name):
    vector = np.asarray(value)
    if vector.shape != (size,):
        raise ValueError(f"{name} must return a vector with shape {(size,)}.")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} returned a nonfinite vector.")
    return vector


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
    initial_subspace=None,
    recycle_out=None,
    hamiltonian_actions=None,
    metric_actions=None,
    preconditioner=None,
    block_size: int = 1,
    verification_hamiltonian_action=None,
    verification_metric_action=None,
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
    initial_subspace
        Optional vectors recycled from a nearby solve. Their operator images
        are recomputed for the current local problem before use.
    recycle_out
        Optional mutable list replaced by a compact pair of vectors suitable
        for a subsequent solve of the same dimension.
    hamiltonian_actions, metric_actions
        Optional native batched actions accepting arrays of shape
        ``(batch, size)``. These are used when a recycled subspace seeds the
        solve; scalar actions remain the reference fallback.
    preconditioner
        Optional callable ``preconditioner(residual, eigenvalue)``. It should
        approximate ``(H - eigenvalue*N)**-1 residual`` and is applied before
        orthogonalization. The final eigenpair is always verified with fresh
        unpreconditioned operator actions.
    block_size
        Number of correction directions appended together. Values above one
        use the native batched actions and supplement the preconditioned
        residual with deterministic orthogonal probes.
    verification_hamiltonian_action, verification_metric_action
        Optional full-precision actions used for the final Rayleigh quotient
        and residual. If a reduced-precision solve does not satisfy the
        requested tolerance, Davidson automatically refines from that vector
        with these actions.

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
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    block_size = min(block_size, max_subspace - 1, size)
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("preconditioner must be callable or None.")

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

    def apply_many(vectors, action, scalar_action, *, name, counter):
        nonlocal hamiltonian_matvecs, metric_matvecs
        vectors = np.asarray(vectors)
        if action is None or vectors.shape[0] == 1:
            return np.stack([scalar_action(vector) for vector in vectors])
        result = np.asarray(action(vectors))
        if result.shape != vectors.shape:
            raise ValueError(
                f"{name} must return an array with shape {vectors.shape}."
            )
        if np.any(~np.isfinite(result)):
            raise ValueError(f"{name} returned nonfinite vectors.")
        if counter == "hamiltonian":
            hamiltonian_matvecs += vectors.shape[0]
        else:
            metric_matvecs += vectors.shape[0]
        return result

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

    def append_basis_batch(vectors):
        accepted = []
        for vector in vectors:
            vector = np.asarray(vector)
            if vector.shape != (size,) or np.any(~np.isfinite(vector)):
                continue
            vector = _orthogonalize(
                vector,
                [*basis, *accepted],
                tolerance=breakdown_tolerance,
            )
            if vector is not None:
                accepted.append(vector)
            if len(basis) + len(accepted) >= max_subspace:
                break
        if not accepted:
            return 0
        stacked = np.stack(accepted)
        h_stacked = apply_many(
            stacked,
            hamiltonian_actions,
            apply_h,
            name="hamiltonian_actions",
            counter="hamiltonian",
        )
        n_stacked = apply_many(
            stacked,
            metric_actions,
            apply_n,
            name="metric_actions",
            counter="metric",
        )
        basis.extend(accepted)
        hamiltonian_basis.extend(h_stacked)
        metric_basis.extend(n_stacked)
        return len(accepted)

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

    def precondition(vector, eigenvalue):
        if preconditioner is None:
            return vector
        result = np.asarray(preconditioner(vector, eigenvalue))
        if result.shape != (size,):
            raise ValueError(
                f"preconditioner must return a vector with shape {(size,)}."
            )
        if np.any(~np.isfinite(result)):
            raise ValueError("preconditioner returned a nonfinite vector.")
        return result

    if _orthogonalize(
        initial_vector,
        (),
        tolerance=breakdown_tolerance,
    ) is None:
        raise ValueError("initial_vector is numerically zero.")
    seed_vectors = [initial_vector]
    if initial_subspace is not None:
        seed_vectors.extend(initial_subspace)
    append_basis_batch(seed_vectors)

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
            precondition(current_residual, current_energy),
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
                precondition(current_residual, current_energy),
                basis,
                tolerance=breakdown_tolerance,
            )
            if correction is None:
                correction = _orthogonalize(
                    random_probe(),
                    basis,
                    tolerance=breakdown_tolerance,
                )
        corrections = [] if correction is None else [correction]
        for _probe in range(block_size - len(corrections)):
            probe = _orthogonalize(
                precondition(random_probe(), current_energy),
                [*basis, *corrections],
                tolerance=breakdown_tolerance,
            )
            if probe is not None:
                corrections.append(probe)
        if not corrections or append_basis_batch(corrections) == 0:
            message = "Davidson restart failed to produce a new direction"
            break

    if current_vector is None:
        raise ValueError(message)

    verify_h = (
        apply_h
        if verification_hamiltonian_action is None
        else lambda vector: _as_finite_vector(
            verification_hamiltonian_action(vector),
            size,
            name="verification_hamiltonian_action",
        )
    )
    verify_n = (
        apply_n
        if verification_metric_action is None
        else lambda vector: _as_finite_vector(
            verification_metric_action(vector),
            size,
            name="verification_metric_action",
        )
    )
    h_final = verify_h(current_vector)
    n_final = verify_n(current_vector)
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
    if (
        not converged
        and (
            verification_hamiltonian_action is not None
            or verification_metric_action is not None
        )
    ):
        refinement_subspace = [current_residual]
        return lowest_generalized_davidson(
            verification_hamiltonian_action or hamiltonian_action,
            verification_metric_action or metric_action,
            current_vector,
            tol=tol,
            atol=atol,
            metric_tol=metric_tol,
            maxiter=maxiter,
            max_subspace=max_subspace,
            random_seed=random_seed,
            initial_subspace=refinement_subspace,
            recycle_out=recycle_out,
            preconditioner=preconditioner,
            block_size=block_size,
        )
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
    if recycle_out is not None:
        retained_direction = current_vector / np.linalg.norm(current_vector)
        residual_direction = _orthogonalize(
            current_residual,
            [retained_direction],
            tolerance=breakdown_tolerance,
        )
        # Retaining the converged Ritz vector is more useful than retaining
        # only its nearly-zero residual. Operator images are deliberately
        # recomputed for the next local environment, so this remains a safe
        # Krylov seed rather than a stale-action cache.
        recycle_out[:] = [np.array(retained_direction, copy=True)]
        if residual_direction is not None:
            recycle_out.append(np.array(residual_direction, copy=True))
    return current_energy, current_vector, diagnostics


__all__ = ["DavidsonDiagnostics", "lowest_generalized_davidson"]
