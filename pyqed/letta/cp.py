"""Small NumPy implementation of canonical-polyadic tensor decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _reconstruct(weights, factors) -> np.ndarray:
    factors = tuple(np.asarray(factor) for factor in factors)
    weights = np.asarray(weights)
    shape = tuple(factor.shape[0] for factor in factors)
    dtype = np.result_type(weights.dtype, *[factor.dtype for factor in factors])
    tensor = np.zeros(shape, dtype=dtype)
    for component, weight in enumerate(weights):
        term = np.asarray(weight, dtype=dtype)
        for factor in factors:
            term = np.multiply.outer(term, factor[:, component])
        tensor += term
    return tensor


@dataclass(frozen=True)
class CPDecomposition:
    """A canonical-polyadic decomposition produced by :func:`cp_als`.

    ``factors[mode]`` has shape ``(tensor.shape[mode], rank)``.  The columns
    are normalized when their corresponding weight is nonzero.
    """

    weights: np.ndarray
    factors: tuple[np.ndarray, ...]
    relative_error: float
    n_iter: int
    converged: bool

    @property
    def rank(self) -> int:
        return int(self.weights.size)

    def reconstruct(self) -> np.ndarray:
        """Return the dense tensor represented by this decomposition."""
        return _reconstruct(self.weights, self.factors)


def _random_columns(rng, size: int, count: int, *, complex_data: bool) -> np.ndarray:
    if count == 0:
        dtype = complex if complex_data else float
        return np.empty((size, 0), dtype=dtype)
    columns = rng.normal(size=(size, count))
    if complex_data:
        columns = columns + 1.0j * rng.normal(size=(size, count))
    norms = np.linalg.norm(columns, axis=0)
    zero = norms == 0.0
    if np.any(zero):  # pragma: no cover - a Gaussian column is almost surely nonzero
        columns[:, zero] = 0.0
        columns[0, zero] = 1.0
        norms[zero] = 1.0
    return columns / norms


def _svd_initial_factors(tensor: np.ndarray, rank: int, rng) -> list[np.ndarray]:
    factors = []
    complex_data = np.iscomplexobj(tensor)
    for mode, dim in enumerate(tensor.shape):
        unfolding = np.moveaxis(tensor, mode, 0).reshape(dim, -1)
        left, _singular_values, _right = np.linalg.svd(unfolding, full_matrices=False)
        n_svd = min(rank, left.shape[1])
        factor = np.empty((dim, rank), dtype=tensor.dtype)
        factor[:, :n_svd] = left[:, :n_svd]
        if n_svd < rank:
            factor[:, n_svd:] = _random_columns(
                rng,
                dim,
                rank - n_svd,
                complex_data=complex_data,
            )
        factors.append(factor)
    return factors


def _khatri_rao(factors: list[np.ndarray], rank: int) -> np.ndarray:
    """Return the column-wise Kronecker product in C unfolding order."""
    if not factors:
        return np.ones((1, rank))
    product = factors[0]
    for factor in factors[1:]:
        product = np.einsum("ir,jr->ijr", product, factor, optimize=True).reshape(
            product.shape[0] * factor.shape[0],
            rank,
        )
    return product


def _balance_factors(factors: list[np.ndarray]) -> None:
    """Balance component norms across modes without changing the tensor."""
    order = len(factors)
    rank = factors[0].shape[1]
    for component in range(rank):
        norms = np.asarray(
            [np.linalg.norm(factor[:, component]) for factor in factors],
            dtype=float,
        )
        if np.any(norms == 0.0):
            continue
        common = float(np.exp(np.mean(np.log(norms))))
        for factor, norm in zip(factors, norms):
            factor[:, component] *= common / norm


def _normalized_decomposition(
    factors: list[np.ndarray],
    *,
    relative_error: float,
    n_iter: int,
    converged: bool,
) -> CPDecomposition:
    normalized = tuple(factor.copy() for factor in factors)
    rank = normalized[0].shape[1]
    weights = np.ones(rank, dtype=float)
    for component in range(rank):
        for factor in normalized:
            norm = float(np.linalg.norm(factor[:, component]))
            if norm == 0.0:
                weights[component] = 0.0
                break
            factor[:, component] /= norm
            weights[component] *= norm
    return CPDecomposition(
        weights=weights,
        factors=normalized,
        relative_error=float(relative_error),
        n_iter=int(n_iter),
        converged=bool(converged),
    )


def _vector_decomposition(tensor: np.ndarray, rank: int) -> CPDecomposition:
    factor = np.zeros((tensor.shape[0], rank), dtype=tensor.dtype)
    norm = float(np.linalg.norm(tensor))
    weights = np.zeros(rank, dtype=float)
    if norm:
        factor[:, 0] = tensor / norm
        weights[0] = norm
    else:
        factor[0, 0] = 1.0
    return CPDecomposition(
        weights=weights,
        factors=(factor,),
        relative_error=0.0,
        n_iter=0,
        converged=True,
    )


def _matrix_decomposition(tensor: np.ndarray, rank: int) -> CPDecomposition:
    left, singular_values, right_h = np.linalg.svd(tensor, full_matrices=False)
    effective_rank = min(rank, singular_values.size)
    factors = (
        np.zeros((tensor.shape[0], rank), dtype=tensor.dtype),
        np.zeros((tensor.shape[1], rank), dtype=tensor.dtype),
    )
    weights = np.zeros(rank, dtype=float)
    factors[0][:, :effective_rank] = left[:, :effective_rank]
    factors[1][:, :effective_rank] = right_h[:effective_rank, :].T
    weights[:effective_rank] = singular_values[:effective_rank]
    reconstruction = _reconstruct(weights, factors)
    norm = float(np.linalg.norm(tensor))
    error = 0.0 if norm == 0.0 else float(np.linalg.norm(tensor - reconstruction) / norm)
    return CPDecomposition(
        weights=weights,
        factors=factors,
        relative_error=error,
        n_iter=0,
        converged=True,
    )


def cp_als(
    tensor,
    rank: int,
    *,
    max_iter: int = 200,
    tol: float = 1.0e-10,
    seed: int | None = 0,
) -> CPDecomposition:
    """Approximate an arbitrary-order tensor by CP alternating least squares.

    Parameters
    ----------
    tensor
        Real or complex tensor with at least one axis.
    rank
        Number of rank-one outer products retained.
    max_iter
        Maximum number of complete ALS sweeps for tensors of order three or
        greater.  Vectors and matrices are handled exactly by normalization
        and the singular-value decomposition, respectively.
    tol
        Relative-error change used as the convergence threshold.
    seed
        Seed used for deterministic initialization when an unfolding does not
        provide enough singular vectors.

    Returns
    -------
    CPDecomposition
        Normalized component weights and factors together with reconstruction
        diagnostics.  CP-ALS is nonconvex for order-three and higher tensors,
        so the returned solution need not be globally optimal.
    """
    array = np.asarray(tensor)
    if array.ndim < 1:
        raise ValueError("tensor must have order at least one.")
    if any(dim < 1 for dim in array.shape):
        raise ValueError("tensor dimensions must be positive.")
    if not np.all(np.isfinite(array)):
        raise ValueError("tensor must contain only finite values.")
    rank = int(rank)
    max_iter = int(max_iter)
    tol = float(tol)
    if rank < 1:
        raise ValueError("rank must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative.")

    dtype = np.result_type(array.dtype, np.float64)
    array = array.astype(dtype, copy=False)
    if array.ndim == 1:
        return _vector_decomposition(array, rank)
    if array.ndim == 2:
        return _matrix_decomposition(array, rank)

    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return CPDecomposition(
            weights=np.zeros(rank, dtype=float),
            factors=tuple(
                np.zeros((dim, rank), dtype=array.dtype) for dim in array.shape
            ),
            relative_error=0.0,
            n_iter=0,
            converged=True,
        )
    rng = np.random.default_rng(seed)
    factors = _svd_initial_factors(array, rank, rng)
    _balance_factors(factors)

    initial = _reconstruct(np.ones(rank), factors)
    initial_error = 0.0 if norm == 0.0 else float(np.linalg.norm(array - initial) / norm)
    best_error = initial_error
    best_factors = [factor.copy() for factor in factors]
    if initial_error <= tol:
        return _normalized_decomposition(
            best_factors,
            relative_error=best_error,
            n_iter=0,
            converged=True,
        )

    previous_error = initial_error
    converged = False
    completed = 0
    for iteration in range(1, max_iter + 1):
        for mode, dim in enumerate(array.shape):
            other_factors = [
                factor
                for other_mode, factor in enumerate(factors)
                if other_mode != mode
            ]
            khatri_rao = _khatri_rao(other_factors, rank)
            unfolding = np.moveaxis(array, mode, 0).reshape(dim, -1)
            gram = khatri_rao.T @ khatri_rao.conj()
            factors[mode] = (
                unfolding
                @ khatri_rao.conj()
                @ np.linalg.pinv(gram, rcond=64.0 * np.finfo(float).eps)
            )
        _balance_factors(factors)
        reconstruction = _reconstruct(np.ones(rank), factors)
        error = float(np.linalg.norm(array - reconstruction) / norm)
        completed = iteration
        if error < best_error:
            best_error = error
            best_factors = [factor.copy() for factor in factors]
        if abs(previous_error - error) <= tol * max(1.0, previous_error):
            converged = True
            break
        previous_error = error

    return _normalized_decomposition(
        best_factors,
        relative_error=best_error,
        n_iter=completed,
        converged=converged,
    )


__all__ = ["CPDecomposition", "cp_als"]
