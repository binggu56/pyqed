"""Continuous matrix product states.

This module contains a small NumPy-only cMPS core for translation-invariant
one-dimensional continuum problems.  It currently targets the real
left-canonical bosonic form used by the Lieb-Liniger examples, while keeping
the transfer machinery complex-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ContinuousMPS",
    "CMPS",
    "canonical_parameter_size",
    "fit_exponential_kernel_nonlinear",
    "fit_exponential_kernel_prony",
    "pack_canonical_parameters",
    "skew_pairs",
    "softened_yukawa_kernel",
    "unpack_canonical_parameters",
]


def skew_pairs(dim: int):
    """Return upper-triangular index pairs for a real skew generator."""
    dim = int(dim)
    if dim < 1:
        raise ValueError("bond dimension must be positive.")
    return [(row, col) for row in range(dim) for col in range(row + 1, dim)]


def canonical_parameter_size(dim: int, num_fields: int = 1):
    """Number of real parameters in the left-canonical ``A, R`` chart."""
    dim = int(dim)
    num_fields = int(num_fields)
    if num_fields < 1:
        raise ValueError("num_fields must be positive.")
    return len(skew_pairs(dim)) + num_fields * dim * dim


def _normalize_r_ops(r_ops, *, bond_dim: int | None = None):
    arr = np.asarray(r_ops)
    if arr.ndim == 2:
        ops = (arr,)
    elif arr.ndim == 3:
        ops = tuple(arr[index] for index in range(arr.shape[0]))
    else:
        try:
            ops = tuple(np.asarray(op) for op in r_ops)
        except TypeError as exc:
            raise ValueError("r_ops must be a square matrix or a sequence of square matrices.") from exc
    if not ops:
        raise ValueError("at least one R matrix is required.")
    dim = int(bond_dim) if bond_dim is not None else int(ops[0].shape[0])
    normalized = []
    for op in ops:
        matrix = np.asarray(op)
        if matrix.shape != (dim, dim):
            raise ValueError("all R matrices must have shape (bond_dim, bond_dim).")
        normalized.append(matrix)
    return tuple(normalized)


def _skew_from_values(values, dim: int):
    values = np.asarray(values)
    pairs = skew_pairs(dim)
    if values.size != len(pairs):
        raise ValueError(f"skew parameter size {values.size} does not match {len(pairs)}.")
    dtype = np.result_type(values.dtype, float)
    matrix = np.zeros((dim, dim), dtype=dtype)
    for value, (row, col) in zip(values.ravel(), pairs):
        matrix[row, col] = value
        matrix[col, row] = -np.conj(value)
    return matrix


def _skew_values(generator, dim: int):
    arr = np.asarray(generator)
    pairs = skew_pairs(dim)
    if arr.shape == (dim, dim):
        return np.asarray([arr[row, col] for row, col in pairs], dtype=arr.dtype)
    if arr.size == len(pairs):
        return np.asarray(arr, dtype=arr.dtype).ravel()
    raise ValueError("skew generator must be a square matrix or an upper-triangular value vector.")


def pack_canonical_parameters(a_skew, r_ops):
    r"""Pack a real left-canonical cMPS chart.

    The chart is

    $$
    Q = A - \frac{1}{2}\sum_i R_i^\dagger R_i,\qquad A^\dagger=-A.
    $$

    For the real Lieb-Liniger examples, ``A`` is skew-symmetric and there is
    one field matrix ``R``.
    """
    ops = _normalize_r_ops(r_ops)
    dim = ops[0].shape[0]
    values = _skew_values(a_skew, dim)
    return np.concatenate([np.asarray(values).ravel(), *(np.asarray(op).ravel() for op in ops)])


def unpack_canonical_parameters(theta, bond_dim: int, num_fields: int = 1):
    """Return ``(Q, R_ops, A)`` from a packed left-canonical chart."""
    dim = int(bond_dim)
    num_fields = int(num_fields)
    theta = np.asarray(theta)
    expected = canonical_parameter_size(dim, num_fields)
    if theta.size != expected:
        raise ValueError(f"theta size {theta.size} does not match canonical cMPS size {expected}.")
    pairs = skew_pairs(dim)
    a = _skew_from_values(theta[: len(pairs)], dim)
    offset = len(pairs)
    r_ops = []
    for _ in range(num_fields):
        r_ops.append(theta[offset : offset + dim * dim].reshape(dim, dim))
        offset += dim * dim
    q = a - 0.5 * sum(r.conj().T @ r for r in r_ops)
    return q, tuple(r_ops), a


def _real_if_close(value):
    value = np.real_if_close(value)
    if np.ndim(value) == 0:
        return value.item()
    return value


def _distance_array(distances):
    arr = np.asarray(distances, dtype=float)
    scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)
    if np.any(arr < -1.0e-14):
        raise ValueError("cMPS correlation distances must be non-negative.")
    arr = np.maximum(arr, 0.0)
    return arr, scalar


def _dominant_sparse_biorthogonal_pair(
    matrix,
    initial,
    *,
    tolerance: float,
    maxiter: int,
    label: str,
):
    """Return a validated Perron pair of a sparse cMPS transfer generator."""
    from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigs

    matrix = matrix.tocsc()
    adjoint = matrix.conj().T.tocsc()
    size = int(matrix.shape[0])
    initial = np.asarray(initial, dtype=np.complex128)
    requested_tolerance = float(tolerance)
    strict_tolerance = min(requested_tolerance, 1.0e-11)
    attempts = [(requested_tolerance, min(4, size - 2))]
    strict_attempt = (strict_tolerance, min(8, size - 2))
    if strict_attempt != attempts[0]:
        attempts.append(strict_attempt)

    last_error = None
    for attempt_tolerance, count in attempts:
        try:
            right_values, right_vectors = eigs(
                matrix,
                k=count,
                which="LR",
                v0=initial,
                tol=attempt_tolerance,
                maxiter=maxiter,
            )
            perron_tolerance = max(100.0 * attempt_tolerance, 1.0e-10)
            real_candidates = np.flatnonzero(
                np.abs(np.imag(right_values))
                <= perron_tolerance * (1.0 + np.abs(np.real(right_values)))
            )
            if real_candidates.size == 0:
                last_error = "Arnoldi did not return a real Perron-root candidate"
                continue
            right_index = int(
                real_candidates[
                    np.argmax(np.real(right_values[real_candidates]))
                ]
            )
            eigenvalue = right_values[right_index]
            spectral_scale = 1.0 + abs(eigenvalue)
            if np.real(eigenvalue) < np.max(np.real(right_values)) - perron_tolerance * spectral_scale:
                last_error = "a complex Ritz value outran the real Perron-root candidate"
                continue
            right = right_vectors[:, right_index]

            left_values, left_vectors = eigs(
                adjoint,
                k=count,
                which="LR",
                v0=initial,
                tol=attempt_tolerance,
                maxiter=maxiter,
            )
            left_index = int(
                np.argmin(np.abs(left_values - eigenvalue.conjugate()))
            )
            left = left_vectors[:, left_index]
            right_residual = np.linalg.norm(matrix @ right - eigenvalue * right)
            left_residual = np.linalg.norm(
                adjoint @ left - eigenvalue.conjugate() * left
            )
            residual_limit = max(100.0 * attempt_tolerance, 1.0e-10) * spectral_scale
            overlap = np.vdot(left, right)
            if left_residual > residual_limit or abs(overlap) <= 1.0e-10:
                spectral_shift = max(1.0e-8, 10.0 * attempt_tolerance) * spectral_scale
                shifted_values, shifted_vectors = eigs(
                    adjoint,
                    k=1,
                    sigma=eigenvalue.conjugate() + spectral_shift,
                    which="LM",
                    v0=initial,
                    tol=min(attempt_tolerance, 1.0e-11),
                    maxiter=maxiter,
                )
                left = shifted_vectors[:, 0]
                left_residual = np.linalg.norm(
                    adjoint @ left - eigenvalue.conjugate() * left
                )
                overlap = np.vdot(left, right)
            if right_residual > residual_limit or left_residual > residual_limit:
                last_error = (
                    "right/left Arnoldi residuals exceed the requested accuracy"
                )
                continue
            if abs(overlap) <= 1.0e-10:
                last_error = "dominant left/right environments are ill-conditioned"
                continue
            return eigenvalue, left, right / overlap
        except (ArpackError, ArpackNoConvergence) as exc:
            last_error = str(exc)

    detail = f" ({last_error})" if last_error else ""
    raise FloatingPointError(f"{label} Arnoldi iteration did not converge{detail}.")


def _exponential_kernel_terms(decay_rates, strengths=None):
    rates = np.atleast_1d(np.asarray(decay_rates, dtype=float))
    if strengths is None:
        weights = np.ones_like(rates)
    else:
        weights = np.atleast_1d(np.asarray(strengths, dtype=float))
    if rates.ndim != 1 or weights.ndim != 1 or rates.size != weights.size:
        raise ValueError("decay_rates and strengths must be one-dimensional arrays of the same size.")
    if rates.size < 1:
        raise ValueError("at least one exponential kernel term is required.")
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("all decay_rates must be finite and positive.")
    if np.any(~np.isfinite(weights)):
        raise ValueError("all strengths must be finite.")
    return rates, weights


def softened_yukawa_kernel(distances, *, strength: float = 1.0, screening: float = 0.0, softening: float = 1.0):
    r"""Return ``strength * exp(-screening*r) / sqrt(r^2 + softening^2)``."""
    r = np.asarray(distances, dtype=float)
    softening = float(softening)
    if softening <= 0.0:
        raise ValueError("softening must be positive.")
    screening = float(screening)
    if screening < 0.0:
        raise ValueError("screening must be non-negative.")
    return float(strength) * np.exp(-screening * np.abs(r)) / np.sqrt(r * r + softening * softening)


def fit_exponential_kernel_prony(distances, values, rank: int, *, rcond=None, real_tol: float = 1.0e-8):
    r"""Fit ``values(r)`` as ``sum_a coeff_a exp(-rate_a r)`` by Prony.

    The distances must be a positive, equally spaced grid.  This is the
    continuum version of the spatial Prony fit used in the GDVR density-kernel
    MPO path.
    """
    distances = np.asarray(distances, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=complex).reshape(-1)
    if distances.shape != values.shape:
        raise ValueError("distances and values must have the same shape.")
    if distances.size < 2:
        raise ValueError("Need at least two samples for a Prony fit.")
    if np.any(~np.isfinite(distances)) or np.any(distances <= 0.0):
        raise ValueError("distances must be finite and positive.")
    steps = np.diff(distances)
    if steps.size and not np.allclose(steps, steps[0], rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("Prony kernel fitting expects equally spaced distances.")
    dx = float(steps[0]) if steps.size else float(distances[0])
    offsets = distances / dx

    rank = int(rank)
    if not (1 <= rank < values.size):
        raise ValueError("rank must satisfy 1 <= rank < len(values).")
    rows = values.size - rank
    predictor = np.zeros((rows, rank), dtype=complex)
    rhs = np.zeros(rows, dtype=complex)
    for row in range(rows):
        predictor[row] = values[row : row + rank]
        rhs[row] = -values[row + rank]
    recurrence, *_ = np.linalg.lstsq(predictor, rhs, rcond=rcond)
    polynomial = np.concatenate(([1.0 + 0.0j], recurrence[::-1]))
    lambdas = np.roots(polynomial)

    stable = np.abs(lambdas) < 1.0 - 1.0e-12
    decaying_lambdas = lambdas[stable]
    if decaying_lambdas.size == 0:
        raise ValueError("Prony fit did not produce decaying exponentials.")
    vandermonde = decaying_lambdas[None, :] ** offsets[:, None]
    coeffs, *_ = np.linalg.lstsq(vandermonde, values, rcond=rcond)
    rates = -np.log(decaying_lambdas) / dx

    nearly_real = (np.abs(np.imag(rates)) <= float(real_tol)) & (
        np.abs(np.imag(coeffs)) <= float(real_tol) * np.maximum(1.0, np.abs(coeffs))
    )
    real_positive_rate = nearly_real & (np.real(rates) > 0.0)
    if not np.all(real_positive_rate):
        rates = rates[real_positive_rate]
        coeffs = coeffs[real_positive_rate]
        if rates.size == 0:
            raise ValueError("Prony fit did not produce real positive decay rates.")
    rates = np.real(rates)
    coeffs = np.real(coeffs)
    order = np.argsort(rates)
    rates = rates[order]
    coeffs = coeffs[order]

    fitted = np.exp(-distances[:, None] * rates[None, :]) @ coeffs
    residual = fitted - np.real_if_close(values)
    denom = float(np.linalg.norm(values))
    rel_error = float(np.linalg.norm(residual) / denom) if denom > 0.0 else float(np.linalg.norm(residual))
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    max_rel = float(np.max(np.abs(residual) / np.maximum(np.abs(values), 1.0e-30)))
    return {
        "decay_rates": rates,
        "strengths": coeffs,
        "fitted": np.asarray(fitted, dtype=float),
        "residual": np.asarray(residual, dtype=float),
        "rel_error": rel_error,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "lambdas": np.exp(-rates * dx),
        "dx": dx,
    }


def fit_exponential_kernel_nonlinear(
    distances,
    values,
    rank: int,
    *,
    relative: bool = True,
    rate_bounds=None,
    starts: int = 3,
    max_nfev: int = 5000,
    amplitude_regularization: float = 0.0,
    rate_offset: float = 0.0,
):
    r"""Fit a real exponential sum by nonlinear least squares.

    The fitted model is

    $$
    f(r)=\sum_{a=1}^{R} c_a e^{-\lambda_a r},\qquad \lambda_a>0.
    $$

    All requested terms are retained while their real amplitudes and
    logarithmic decay rates are jointly refined.  A known common exponential
    factor can be supplied through ``rate_offset``; the returned rates then
    include that offset.  Relative residuals are used by default so that the
    long-range tail is not overwhelmed by the larger short-range values.
    """
    from scipy.optimize import least_squares

    distances = np.asarray(distances, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if distances.shape != values.shape:
        raise ValueError("distances and values must have the same shape.")
    if distances.size < 2:
        raise ValueError("at least two samples are required.")
    if np.any(~np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("distances must be finite and non-negative.")
    if np.any(~np.isfinite(values)):
        raise ValueError("values must be finite.")
    rank = int(rank)
    starts = int(starts)
    if rank < 1:
        raise ValueError("rank must be positive.")
    if starts < 1:
        raise ValueError("starts must be positive.")
    regularization = float(amplitude_regularization)
    if regularization < 0.0:
        raise ValueError("amplitude_regularization must be non-negative.")
    rate_offset = float(rate_offset)
    if not np.isfinite(rate_offset) or rate_offset < 0.0:
        raise ValueError("rate_offset must be finite and non-negative.")

    value_scale = float(np.max(np.abs(values)))
    if value_scale == 0.0:
        value_scale = 1.0
    scaled_values = values / value_scale

    positive_distances = distances[distances > 0.0]
    minimum_distance = (
        float(np.min(positive_distances)) if positive_distances.size else 1.0
    )
    maximum_distance = max(float(np.max(distances)), minimum_distance)
    if rate_bounds is None:
        lower_rate = max(1.0e-8, 0.1 / maximum_distance)
        upper_rate = max(10.0 * lower_rate, 5.0 / minimum_distance)
    else:
        lower_rate, upper_rate = map(float, rate_bounds)
        if not (0.0 < lower_rate < upper_rate):
            raise ValueError("rate_bounds must satisfy 0 < lower < upper.")

    if relative:
        floor = max(np.max(np.abs(scaled_values)) * 1.0e-12, np.finfo(float).tiny)
        residual_weights = 1.0 / np.maximum(np.abs(scaled_values), floor)
    else:
        residual_weights = np.ones_like(values)
    sqrt_regularization = np.sqrt(regularization)

    lower_bounds = np.concatenate(
        [np.full(rank, -np.inf), np.full(rank, np.log(lower_rate))]
    )
    upper_bounds = np.concatenate(
        [np.full(rank, np.inf), np.full(rank, np.log(upper_rate))]
    )

    def residual_jacobian(parameters):
        amplitudes = parameters[:rank]
        free_rates = np.exp(parameters[rank:])
        exponentials = np.exp(-distances[:, None] * (free_rates + rate_offset))
        residual = residual_weights * (exponentials @ amplitudes - scaled_values)
        amplitude_jacobian = residual_weights[:, None] * exponentials
        rate_jacobian = (
            residual_weights[:, None]
            * exponentials
            * (-distances[:, None] * free_rates)
            * amplitudes
        )
        jacobian = np.concatenate([amplitude_jacobian, rate_jacobian], axis=1)
        if regularization:
            residual = np.concatenate(
                [residual, sqrt_regularization * amplitudes]
            )
            regularization_jacobian = np.zeros((rank, 2 * rank), dtype=float)
            regularization_jacobian[:, :rank] = sqrt_regularization * np.eye(rank)
            jacobian = np.vstack([jacobian, regularization_jacobian])
        return residual, jacobian

    best = None
    logarithmic_span = np.log(upper_rate / lower_rate)
    for start in range(starts):
        contraction = 0.12 * start
        start_lower = lower_rate * np.exp(contraction * logarithmic_span)
        start_upper = upper_rate * np.exp(-contraction * logarithmic_span)
        if start_lower >= start_upper:
            start_lower, start_upper = lower_rate, upper_rate
        free_rates = np.geomspace(start_lower, start_upper, rank)
        exponentials = np.exp(
            -distances[:, None] * (free_rates + rate_offset)
        )
        weighted_matrix = residual_weights[:, None] * exponentials
        weighted_values = residual_weights * scaled_values
        if regularization:
            weighted_matrix = np.vstack(
                [weighted_matrix, sqrt_regularization * np.eye(rank)]
            )
            weighted_values = np.concatenate([weighted_values, np.zeros(rank)])
        amplitudes = np.linalg.lstsq(
            weighted_matrix,
            weighted_values,
            rcond=1.0e-13,
        )[0]
        initial = np.concatenate([amplitudes, np.log(free_rates)])
        result = least_squares(
            lambda parameters: residual_jacobian(parameters)[0],
            initial,
            jac=lambda parameters: residual_jacobian(parameters)[1],
            bounds=(lower_bounds, upper_bounds),
            max_nfev=int(max_nfev),
            ftol=1.0e-13,
            xtol=1.0e-13,
            gtol=1.0e-13,
        )
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result

    amplitudes = value_scale * np.asarray(best.x[:rank], dtype=float)
    rates = rate_offset + np.exp(np.asarray(best.x[rank:], dtype=float))
    order = np.argsort(rates)
    rates = rates[order]
    amplitudes = amplitudes[order]
    fitted = np.exp(-distances[:, None] * rates) @ amplitudes
    residual = fitted - values
    weighted_residual = residual_weights * (residual / value_scale)
    denominator = np.linalg.norm(values)
    rel_error = (
        np.linalg.norm(residual) / denominator
        if denominator
        else np.linalg.norm(residual)
    )
    return {
        "decay_rates": rates,
        "strengths": amplitudes,
        "fitted": np.asarray(fitted, dtype=float),
        "residual": np.asarray(residual, dtype=float),
        "rel_error": float(rel_error),
        "max_abs_error": float(np.max(np.abs(residual))),
        "max_rel_error": float(
            np.max(np.abs(weighted_residual))
        ),
        "relative_rms_error": float(
            np.linalg.norm(weighted_residual) / np.sqrt(values.size)
        ),
        "value_scale": value_scale,
        "rate_offset": rate_offset,
        "nfev": int(best.nfev),
        "success": bool(best.success),
        "message": str(best.message),
        "method": "nonlinear-relative" if relative else "nonlinear-absolute",
    }


@dataclass
class ContinuousMPS:
    r"""Uniform continuous matrix product state.

    Parameters are the cMPS matrices ``Q`` and one or more field matrices
    ``R_i``.  The row-major transfer matrix represents

    $$
    T(X)=QX+XQ^\dagger+\sum_i R_i X R_i^\dagger.
    $$
    """

    q: np.ndarray
    r_ops: tuple[np.ndarray, ...] | np.ndarray
    theta: np.ndarray | None = None
    energy: float | None = None
    density: float | None = None
    kinetic: float | None = None
    contact: float | None = None
    interaction: float | None = None
    raw_density: float | None = None
    scale: float | None = None
    success: bool | None = None
    message: str = ""
    nfev: int = 0
    algorithm: str | None = None
    cletta_base: "ContinuousMPS | None" = None
    cletta_tie_matrices: np.ndarray | None = None
    cletta_decay_rates: np.ndarray | None = None
    cletta_frequencies: np.ndarray | None = None
    cletta_depth: int | None = None
    cletta_parameters: np.ndarray | None = None

    def __post_init__(self):
        self.q = np.asarray(self.q)
        if self.q.ndim != 2 or self.q.shape[0] != self.q.shape[1]:
            raise ValueError("Q must be a square matrix.")
        self.r_ops = _normalize_r_ops(self.r_ops, bond_dim=self.q.shape[0])
        if self.theta is not None:
            self.theta = np.asarray(self.theta, dtype=float)

    @property
    def bond_dim(self):
        return int(self.q.shape[0])

    @property
    def num_fields(self):
        return len(self.r_ops)

    @property
    def r(self):
        if self.num_fields != 1:
            raise ValueError("r is only defined for a single-field cMPS.")
        return self.r_ops[0]

    @classmethod
    def from_canonical_parameters(cls, theta, bond_dim: int, num_fields: int = 1):
        """Build a left-canonical cMPS from packed ``A, R`` parameters."""
        q, r_ops, _a = unpack_canonical_parameters(theta, bond_dim, num_fields)
        return cls(q, r_ops, theta=np.asarray(theta, dtype=float))

    @classmethod
    def random_left_canonical(
        cls,
        bond_dim: int,
        *,
        num_fields: int = 1,
        seed=None,
        scale: float = 0.25,
    ):
        """Random real left-canonical cMPS in the packed ``A, R`` chart."""
        rng = np.random.default_rng(seed)
        theta = cls.random_canonical_parameters(
            bond_dim,
            num_fields=num_fields,
            rng=rng,
            scale=scale,
        )
        return cls.from_canonical_parameters(theta, bond_dim, num_fields)

    @staticmethod
    def random_canonical_parameters(
        bond_dim: int,
        *,
        num_fields: int = 1,
        rng=None,
        seed=None,
        scale: float = 0.25,
    ):
        if rng is None:
            rng = np.random.default_rng(seed)
        bond_dim = int(bond_dim)
        num_fields = int(num_fields)
        a = float(scale) * rng.normal(size=len(skew_pairs(bond_dim)))
        r_ops = float(scale) * rng.normal(size=(num_fields, bond_dim, bond_dim))
        return pack_canonical_parameters(a, r_ops)

    def canonical_parameters(self, *, atol: float = 1.0e-9):
        """Return packed ``A, R`` parameters if this state is left canonical."""
        a = self.q + 0.5 * sum(r.conj().T @ r for r in self.r_ops)
        if np.linalg.norm(a + a.conj().T) > float(atol):
            raise ValueError("Q,R are not in left-canonical form.")
        if np.max(np.abs(np.imag(a))) > float(atol):
            raise ValueError("canonical parameter packing currently expects a real skew generator.")
        if any(np.max(np.abs(np.imag(r))) > float(atol) for r in self.r_ops):
            raise ValueError("canonical parameter packing currently expects real R matrices.")
        return pack_canonical_parameters(np.real(a), [np.real(r) for r in self.r_ops])

    def left_canonical_error(self):
        residual = self.q + self.q.conj().T
        for r in self.r_ops:
            residual = residual + r.conj().T @ r
        return float(np.linalg.norm(residual))

    def canonical_drift(self):
        """Return ``A = Q + 1/2 sum_i R_i^dag R_i``."""
        drift = np.array(self.q, copy=True)
        for r in self.r_ops:
            drift = drift + 0.5 * (r.conj().T @ r)
        return drift

    def lindblad_hamiltonian(self, *, atol: float = 1.0e-9):
        """Return the Lindblad Hamiltonian equivalent in left-canonical gauge.

        If ``Q = A - 1/2 sum_i R_i^dag R_i`` and ``A^dag = -A``, then the
        cMPS transfer generator is the Lindblad Liouvillian with
        ``H = 1j * A`` and collapse operators ``R_i``.
        """
        drift = self.canonical_drift()
        error = np.linalg.norm(drift + drift.conj().T)
        if error > float(atol):
            raise ValueError(f"cMPS is not left canonical; anti-Hermitian drift error is {error:.3e}.")
        hamiltonian = 1j * drift
        return 0.5 * (hamiltonian + hamiltonian.conj().T)

    def lindblad_liouvillian(self, *, atol: float = 1.0e-9, dense: bool = False):
        """Build the equivalent pyqed Lindblad Liouvillian.

        The return value is the sparse matrix produced by
        :func:`pyqed.superoperator.liouvillian`, unless ``dense=True``.
        """
        from pyqed import superoperator

        hamiltonian = self.lindblad_hamiltonian(atol=atol)
        liouvillian = superoperator.liouvillian(hamiltonian, list(self.r_ops))
        if dense and hasattr(liouvillian, "toarray"):
            return liouvillian.toarray()
        return liouvillian

    def to_lindblad_solver(self, *, atol: float = 1.0e-9):
        """Return a :class:`pyqed.superoperator.Lindblad_solver` adapter."""
        from pyqed.superoperator import Lindblad_solver

        return Lindblad_solver(self.lindblad_hamiltonian(atol=atol), c_ops=list(self.r_ops))

    def heom(
        self,
        coupling,
        gamma,
        eta,
        *,
        depth: int = 1,
        mode=None,
        truncation: str = "total",
        weights=None,
        atol: float = 1.0e-9,
    ):
        """Return an HEOM solver for exponential memory on the auxiliary space.

        The zero-tier block is the cMPS auxiliary Lindblad generator, stored on
        the returned solver as its base Liouvillian.
        """
        from pyqed.heom import Bath, HEOM

        bath = Bath.from_exponential_terms(gamma, eta, mode=mode)
        aux = np.asarray(self.lindblad_liouvillian(atol=atol, dense=True), dtype=np.complex128)
        return HEOM(
            system=self.lindblad_hamiltonian(atol=atol),
            system_liouvillian=aux,
            bath=bath,
            coupling=coupling,
            lmax=depth,
            hierarchy_truncation=truncation,
            hierarchy_weights=weights,
        )

    def heom_kernel(
        self,
        coupling,
        gamma,
        eta,
        *,
        depth: int = 1,
        mode=None,
        truncation: str = "total",
        weights=None,
        atol: float = 1.0e-9,
    ):
        """Build an exponential-memory HEOM on top of the auxiliary Lindblad.

        Returns a dictionary containing the base auxiliary Lindblad generator,
        the full HEOM hierarchy generator, and the zero-tier diagonal block.
        The zero-tier block should match the base Lindblad generator; off
        diagonal blocks are the HEOM memory couplings.
        """
        solver = self.heom(
            coupling,
            gamma,
            eta,
            depth=depth,
            mode=mode,
            truncation=truncation,
            weights=weights,
            atol=atol,
        )
        aux = np.asarray(solver.system_liouvillian, dtype=np.complex128)
        liouvillian = solver.hierarchy_liouvillian()
        block_size = aux.shape[0]
        zero = liouvillian[:block_size, :block_size]
        return {
            "solver": solver,
            "aux": aux,
            "L": liouvillian,
            "zero": zero,
            "error": float(np.linalg.norm(zero - aux)),
            "keys": np.array(solver.keys, copy=True),
            "n_ado": int(solver.nmax),
        }

    def heom_contract(
        self,
        distances,
        *,
        coupling,
        gamma,
        eta,
        final_operator,
        initial_operator=None,
        initial_matrix=None,
        rho=None,
        depth: int = 1,
        mode=None,
        truncation: str = "total",
        weights=None,
        method: str = "eig",
        atol: float = 1.0e-9,
    ):
        r"""Contract a zero-tier HEOM two-point object.

        The initial matrix is embedded into the zero ADO, evolved by the HEOM
        hierarchy Liouvillian, and traced from the zero ADO:

        $$
        \mathrm{Tr}_0[O_f e^{x\mathcal{L}_{\rm HEOM}}(X_0)].
        $$
        """
        solver = self.heom(
            coupling,
            gamma,
            eta,
            depth=depth,
            mode=mode,
            truncation=truncation,
            weights=weights,
            atol=atol,
        )
        if initial_matrix is None:
            if initial_operator is None:
                raise ValueError("initial_operator or initial_matrix is required.")
            rho = self.right_fixed_density() if rho is None else np.asarray(rho)
            return solver.correlation_2p_1t(
                rho,
                [final_operator, initial_operator],
                distances,
                method=method,
            )

        return solver.zero_tier_contract(
            initial_matrix,
            final_operator,
            distances,
            method=method,
        )

    def cletta_memory_state(
        self,
        tie_matrices,
        decay_rates,
        *,
        depth: int = 1,
        frequencies=None,
    ):
        r"""Return the explicit infinite-continuum cLETTA memory cMPS.

        The returned state has no real-space grid or finite length.  It is the
        uniform continuum cMPS obtained by replacing the single auxiliary
        field matrix with the finite-depth exponential-memory cLETTA matrices.
        """
        if self.num_fields != 1:
            raise ValueError("cLETTA memory states currently require a single physical field.")
        from .cletta import cletta_multimode_memory_matrices

        ties = np.asarray(tie_matrices)
        if ties.ndim == 2:
            ties = ties[np.newaxis, :, :]
        rates = np.atleast_1d(np.asarray(decay_rates, dtype=float))
        q_memory, r_memory = cletta_multimode_memory_matrices(
            self.q,
            self.r,
            ties,
            rates,
            depth=depth,
            frequencies=frequencies,
        )
        state = ContinuousMPS(q_memory, r_memory)
        state.cletta_base = self
        state.cletta_tie_matrices = np.array(ties, copy=True)
        state.cletta_decay_rates = np.array(rates, copy=True)
        state.cletta_frequencies = None if frequencies is None else np.array(frequencies, copy=True)
        state.cletta_depth = int(depth)
        return state

    def cletta_aux_lindblad_state(
        self,
        tie_matrices,
        decay_rates,
        *,
        cutoff: int = 1,
        frequencies=None,
    ):
        """Return the explicit finite-Fock auxiliary-mode cLETTA state.

        This is the pseudomode/auxiliary-Lindblad representation.  The
        two-sided cLETTA hierarchy is an equivalent contraction form for this
        explicit auxiliary state.
        """
        return self.cletta_memory_state(
            tie_matrices,
            decay_rates,
            depth=cutoff,
            frequencies=frequencies,
        )

    def cletta_lieb_liniger_fixed_density_observables(
        self,
        tie_matrices,
        decay_rates,
        *,
        coupling: float,
        density: float = 1.0,
        depth: int = 1,
        frequencies=None,
    ):
        """Return fixed-density Lieb-Liniger observables for a cLETTA state."""
        state = self.cletta_memory_state(
            tie_matrices,
            decay_rates,
            depth=depth,
            frequencies=frequencies,
        )
        return state.lieb_liniger_fixed_density_observables(
            coupling=coupling,
            density=density,
            canonical=False,
        )

    def cletta_exponential_bose_gas_fixed_density_observables(
        self,
        tie_matrices,
        memory_decay_rates,
        *,
        interaction_decay_rates,
        strengths=None,
        density: float = 1.0,
        depth: int = 1,
        frequencies=None,
        connected: bool = False,
        contact_coupling: float = 0.0,
        contraction_backend: str = "explicit",
        iterative_tolerance: float = 1.0e-9,
        iterative_maxiter: int | None = None,
    ):
        """Return fixed-density exponential-kernel observables for cLETTA."""
        backend = str(contraction_backend).lower().replace("-", "_")
        if backend in {"hierarchy_iterative", "heom_iterative", "matrix_free"}:
            return self._cletta_exponential_bose_gas_iterative_hierarchy_fixed_density_observables(
                tie_matrices,
                memory_decay_rates,
                interaction_decay_rates=interaction_decay_rates,
                strengths=strengths,
                density=density,
                depth=depth,
                frequencies=frequencies,
                connected=connected,
                contact_coupling=contact_coupling,
                tolerance=iterative_tolerance,
                maxiter=iterative_maxiter,
            )
        if backend in {"hierarchy", "heom", "double_hierarchy", "two_sided_heom"}:
            return self._cletta_exponential_bose_gas_hierarchy_fixed_density_observables(
                tie_matrices,
                memory_decay_rates,
                interaction_decay_rates=interaction_decay_rates,
                strengths=strengths,
                density=density,
                depth=depth,
                frequencies=frequencies,
                connected=connected,
                contact_coupling=contact_coupling,
            )
        if backend not in {"explicit", "pseudomode", "aux_lindblad", "auxiliary_lindblad"}:
            raise ValueError("contraction_backend must be 'explicit' or 'hierarchy'.")
        state = self.cletta_memory_state(
            tie_matrices,
            memory_decay_rates,
            depth=depth,
            frequencies=frequencies,
        )
        return state.exponential_bose_gas_fixed_density_observables(
            decay_rates=interaction_decay_rates,
            strengths=strengths,
            density=density,
            canonical=False,
            connected=connected,
            contact_coupling=contact_coupling,
        )

    def _cletta_exponential_bose_gas_hierarchy_fixed_density_observables(
        self,
        tie_matrices,
        memory_decay_rates,
        *,
        interaction_decay_rates,
        strengths=None,
        density: float = 1.0,
        depth: int = 1,
        frequencies=None,
        connected: bool = False,
        contact_coupling: float = 0.0,
    ):
        """Evaluate cLETTA observables with the exact two-sided hierarchy."""
        from .cletta import (
            cletta_memory_fock_keys,
            cletta_multimode_bra_insertion_matrix,
            cletta_multimode_hierarchy_generator,
            cletta_multimode_ket_insertion_matrix,
            cletta_multimode_memory_matrices,
            hierarchy_blocks_to_matrix,
        )

        target_density = float(density)
        contact_coupling = float(contact_coupling)
        if target_density <= 0.0:
            raise ValueError("density must be positive.")
        if not np.isfinite(contact_coupling):
            raise ValueError("contact_coupling must be finite.")
        rates, weights = _exponential_kernel_terms(interaction_decay_rates, strengths)
        memory_rates = np.atleast_1d(np.asarray(memory_decay_rates, dtype=float))
        ties = np.asarray(tie_matrices, dtype=float)
        if ties.ndim == 2:
            ties = ties[np.newaxis, :, :]

        generator = cletta_multimode_hierarchy_generator(
            self.q,
            self.r,
            ties,
            memory_rates,
            depth=depth,
            frequencies=frequencies,
        )
        ket = cletta_multimode_ket_insertion_matrix(
            self.r,
            ties,
            memory_rates,
            depth=depth,
            frequencies=frequencies,
        )
        bra = cletta_multimode_bra_insertion_matrix(
            self.r,
            ties,
            memory_rates,
            depth=depth,
            frequencies=frequencies,
        )
        density_insertion = bra @ ket

        values, right_vectors = np.linalg.eig(generator)
        index = int(np.argmax(np.real(values)))
        eigenvalue = values[index]
        right = right_vectors[:, index]
        left_values, left_vectors = np.linalg.eig(generator.conj().T)
        left_index = int(np.argmin(np.abs(left_values - eigenvalue.conj())))
        left = left_vectors[:, left_index]
        overlap = np.vdot(left, right)
        if abs(overlap) <= 1.0e-12:
            raise FloatingPointError("dominant cLETTA hierarchy environments are nearly orthogonal.")
        right = right / overlap

        raw_density_value = np.vdot(left, density_insertion @ right)
        raw_density = float(np.real(raw_density_value))
        if raw_density <= 0.0:
            raise FloatingPointError("raw cLETTA density must be positive for fixed-density scaling.")
        scale = target_density / raw_density

        memory_dim = len(cletta_memory_fock_keys(len(memory_rates), depth))
        q_memory, r_memory = cletta_multimode_memory_matrices(
            self.q,
            self.r,
            ties,
            memory_rates,
            depth=depth,
            frequencies=frequencies,
        )
        commutator = q_memory @ r_memory - r_memory @ q_memory
        explicit_kinetic_insertion = np.kron(commutator, commutator.conj())
        size = generator.shape[0]
        permutation = np.zeros((size, size), dtype=np.complex128)
        block_shape = (memory_dim, memory_dim, self.bond_dim, self.bond_dim)
        for column in range(size):
            blocks = np.zeros(block_shape, dtype=np.complex128)
            blocks.reshape(-1)[column] = 1.0
            permutation[:, column] = hierarchy_blocks_to_matrix(blocks).reshape(-1)
        kinetic_insertion = permutation.conj().T @ explicit_kinetic_insertion @ permutation
        kinetic = scale**3 * float(np.real(np.vdot(left, kinetic_insertion @ right)))
        pair = r_memory @ r_memory
        explicit_contact_insertion = np.kron(pair, pair.conj())
        contact_insertion = permutation.conj().T @ explicit_contact_insertion @ permutation
        contact = scale**2 * float(np.real(np.vdot(left, contact_insertion @ right)))

        shifted = generator - eigenvalue * np.eye(size, dtype=np.complex128)
        interaction = 0.0
        for rate, weight in zip(rates, weights):
            alpha = float(rate) / scale
            solved = np.linalg.solve(
                shifted - alpha * np.eye(size, dtype=np.complex128),
                density_insertion @ right,
            )
            integral = -np.vdot(left, density_insertion @ solved)
            if connected:
                integral = integral - raw_density_value * raw_density_value / alpha
            interaction += scale * float(weight) * float(np.real(integral))
        energy = kinetic + interaction + contact_coupling * contact
        return {
            "energy_density": float(energy),
            "density": target_density,
            "kinetic": float(kinetic),
            "contact": float(contact),
            "interaction": float(interaction),
            "raw_density": raw_density,
            "scale": float(scale),
        }

    def _cletta_exponential_bose_gas_iterative_hierarchy_fixed_density_observables(
        self,
        tie_matrices,
        memory_decay_rates,
        *,
        interaction_decay_rates,
        strengths=None,
        density: float = 1.0,
        depth: int = 1,
        frequencies=None,
        connected: bool = False,
        contact_coupling: float = 0.0,
        tolerance: float = 1.0e-9,
        maxiter: int | None = None,
    ):
        """Evaluate cLETTA observables with matrix-free Arnoldi and GMRES."""
        from scipy.sparse.linalg import (
            LinearOperator,
            gmres,
        )

        from .cletta import (
            apply_cletta_multimode_bra_insertion,
            apply_cletta_multimode_ket_insertion,
            cletta_memory_fock_keys,
            cletta_multimode_hierarchy_sparse_generator,
            cletta_multimode_memory_matrices,
            hierarchy_blocks_to_matrix,
            matrix_to_hierarchy_blocks,
        )

        target_density = float(density)
        contact_coupling = float(contact_coupling)
        tolerance = float(tolerance)
        if target_density <= 0.0:
            raise ValueError("density must be positive.")
        if not np.isfinite(contact_coupling):
            raise ValueError("contact_coupling must be finite.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("iterative tolerance must be finite and positive.")
        rates, weights = _exponential_kernel_terms(interaction_decay_rates, strengths)
        memory_rates = np.atleast_1d(np.asarray(memory_decay_rates, dtype=float))
        if frequencies is None:
            memory_frequencies = np.zeros_like(memory_rates)
        else:
            memory_frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        ties = np.asarray(tie_matrices, dtype=float)
        if ties.ndim == 2:
            ties = ties[np.newaxis, :, :]

        keys = cletta_memory_fock_keys(len(memory_rates), depth)
        memory_dim = len(keys)
        block_shape = (memory_dim, memory_dim, self.bond_dim, self.bond_dim)
        size = int(np.prod(block_shape))
        if size <= 2:
            values = self._cletta_exponential_bose_gas_hierarchy_fixed_density_observables(
                ties,
                memory_rates,
                interaction_decay_rates=rates,
                strengths=weights,
                density=target_density,
                depth=depth,
                frequencies=memory_frequencies,
                connected=connected,
                contact_coupling=contact_coupling,
            )
            values["hierarchy_size"] = size
            values["gmres_iterations"] = 0
            return values
        iteration_limit = int(maxiter) if maxiter is not None else max(200, 2 * size)

        sparse_generator = cletta_multimode_hierarchy_sparse_generator(
            self.q,
            self.r,
            ties,
            memory_rates,
            depth=depth,
            frequencies=memory_frequencies,
        )
        sparse_adjoint = sparse_generator.conj().T.tocsr()

        def hierarchy_action(vector):
            return sparse_generator @ np.asarray(vector, dtype=np.complex128)

        def hierarchy_adjoint_action(vector):
            return sparse_adjoint @ np.asarray(vector, dtype=np.complex128)

        generator = LinearOperator(
            (size, size),
            matvec=hierarchy_action,
            rmatvec=hierarchy_adjoint_action,
            dtype=np.complex128,
        )
        initial = np.zeros(block_shape, dtype=np.complex128)
        initial[0, 0] = np.eye(self.bond_dim, dtype=np.complex128)
        initial = initial.reshape(-1)
        if size > 1:
            probe = np.arange(1, size + 1, dtype=float)
            initial = initial + 1.0e-6 * probe / np.linalg.norm(probe)
        eigenvalue, left, right = _dominant_sparse_biorthogonal_pair(
            sparse_generator,
            initial,
            tolerance=tolerance,
            maxiter=iteration_limit,
            label="matrix-free cLETTA",
        )

        def ket_action(vector):
            blocks = np.asarray(vector).reshape(block_shape)
            return apply_cletta_multimode_ket_insertion(
                blocks,
                self.r,
                ties,
                memory_rates,
                depth=depth,
                frequencies=memory_frequencies,
            ).reshape(-1)

        def bra_action(vector):
            blocks = np.asarray(vector).reshape(block_shape)
            return apply_cletta_multimode_bra_insertion(
                blocks,
                self.r,
                ties,
                memory_rates,
                depth=depth,
                frequencies=memory_frequencies,
            ).reshape(-1)

        def density_action(vector):
            return bra_action(ket_action(vector))

        density_right = density_action(right)
        raw_density_value = np.vdot(left, density_right)
        raw_density = float(np.real(raw_density_value))
        if raw_density <= 0.0:
            raise FloatingPointError("raw cLETTA density must be positive for fixed-density scaling.")
        scale = target_density / raw_density

        q_memory, r_memory = cletta_multimode_memory_matrices(
            self.q,
            self.r,
            ties,
            memory_rates,
            depth=depth,
            frequencies=memory_frequencies,
        )

        def explicit_insertion(vector, operator):
            blocks = np.asarray(vector).reshape(block_shape)
            matrix = hierarchy_blocks_to_matrix(blocks)
            inserted = operator @ matrix @ operator.conj().T
            return matrix_to_hierarchy_blocks(
                inserted,
                bond_dim=self.bond_dim,
                memory_dim=memory_dim,
            ).reshape(-1)

        commutator = q_memory @ r_memory - r_memory @ q_memory
        kinetic = scale**3 * float(
            np.real(np.vdot(left, explicit_insertion(right, commutator)))
        )
        pair = r_memory @ r_memory
        contact = scale**2 * float(
            np.real(np.vdot(left, explicit_insertion(right, pair)))
        )

        eye_virtual = np.eye(self.bond_dim, dtype=np.complex128)
        base_transfer = (
            np.kron(self.q, eye_virtual)
            + np.kron(eye_virtual, self.q.conj())
            + np.kron(self.r, self.r.conj())
        )
        decay_ket = np.asarray(keys, dtype=float) @ (
            memory_rates + 1.0j * memory_frequencies
        )
        decay_bra = np.asarray(keys, dtype=float) @ (
            memory_rates - 1.0j * memory_frequencies
        )
        block_size = self.bond_dim**2

        def shifted_operator(alpha):
            shift = eigenvalue + float(alpha)
            return LinearOperator(
                (size, size),
                matvec=lambda vector: hierarchy_action(vector) - shift * vector,
                rmatvec=lambda vector: hierarchy_adjoint_action(vector) - shift.conjugate() * vector,
                dtype=np.complex128,
            )

        def block_preconditioner(alpha):
            inverses = np.empty(
                (memory_dim, memory_dim, block_size, block_size),
                dtype=np.complex128,
            )
            identity = np.eye(block_size, dtype=np.complex128)
            for ket_index in range(memory_dim):
                for bra_index in range(memory_dim):
                    shift = (
                        decay_ket[ket_index]
                        + decay_bra[bra_index]
                        + eigenvalue
                        + float(alpha)
                    )
                    inverses[ket_index, bra_index] = np.linalg.inv(
                        base_transfer - shift * identity
                    )

            def apply(vector):
                blocks = np.asarray(vector).reshape(memory_dim, memory_dim, block_size)
                out = np.empty_like(blocks, dtype=np.complex128)
                for ket_index in range(memory_dim):
                    for bra_index in range(memory_dim):
                        out[ket_index, bra_index] = (
                            inverses[ket_index, bra_index] @ blocks[ket_index, bra_index]
                        )
                return out.reshape(-1)

            return LinearOperator((size, size), matvec=apply, dtype=np.complex128)

        interaction = 0.0
        gmres_iterations = 0
        for rate, weight in zip(rates, weights):
            alpha = float(rate) / scale
            counter = [0]

            def count_iteration(_residual):
                counter[0] += 1

            solved, info = gmres(
                shifted_operator(alpha),
                density_right,
                M=block_preconditioner(alpha),
                rtol=tolerance,
                atol=0.0,
                restart=min(80, size),
                maxiter=iteration_limit,
                callback=count_iteration,
                callback_type="pr_norm",
            )
            gmres_iterations += counter[0]
            if info != 0:
                raise FloatingPointError(
                    f"matrix-free cLETTA GMRES did not converge (info={info})."
                )
            integral = -np.vdot(left, density_action(solved))
            if connected:
                integral = integral - raw_density_value * raw_density_value / alpha
            interaction += scale * float(weight) * float(np.real(integral))
        energy = kinetic + interaction + contact_coupling * contact
        return {
            "energy_density": float(energy),
            "density": target_density,
            "kinetic": float(kinetic),
            "contact": float(contact),
            "interaction": float(interaction),
            "raw_density": raw_density,
            "scale": float(scale),
            "hierarchy_size": size,
            "gmres_iterations": int(gmres_iterations),
        }

    def transfer_matrix(self):
        """Dense row-major matrix for the cMPS transfer generator."""
        dtype = np.result_type(self.q, *(r.dtype for r in self.r_ops), np.complex128)
        dim = self.bond_dim
        eye = np.eye(dim, dtype=dtype)
        q = np.asarray(self.q, dtype=dtype)
        transfer = np.kron(q, eye) + np.kron(eye, q.conj())
        for r in self.r_ops:
            r = np.asarray(r, dtype=dtype)
            transfer = transfer + np.kron(r, r.conj())
        return transfer

    def apply_transfer(self, matrix):
        """Apply ``T(X)=QX+XQ^dag+sum R X R^dag`` without forming ``T``."""
        x = np.asarray(matrix)
        out = self.q @ x + x @ self.q.conj().T
        for r in self.r_ops:
            out = out + r @ x @ r.conj().T
        return out

    def right_fixed_density(self, *, trace: float = 1.0):
        """Solve the right fixed density in left-canonical gauge."""
        dim = self.bond_dim
        matrix = self.transfer_matrix().copy()
        rhs = np.zeros(dim * dim, dtype=matrix.dtype)
        matrix[0, :] = np.eye(dim, dtype=matrix.dtype).reshape(-1)
        rhs[0] = trace
        rho = np.linalg.solve(matrix, rhs).reshape(dim, dim)
        rho = 0.5 * (rho + rho.conj().T)
        rho_trace = np.trace(rho)
        if abs(rho_trace) <= 1.0e-14:
            raise FloatingPointError("fixed density has nearly zero trace.")
        return rho * (trace / rho_trace)

    def dominant_fixed_points(self):
        """Return dominant transfer left/right eigenvectors and eigenvalue."""
        transfer = self.transfer_matrix()
        values, vectors = np.linalg.eig(transfer)
        index = int(np.argmax(np.real(values)))
        value = values[index]
        right = vectors[:, index]

        left_values, left_vectors = np.linalg.eig(transfer.conj().T)
        left_index = int(np.argmin(np.abs(left_values - value.conj())))
        left = left_vectors[:, left_index]
        overlap = np.vdot(left, right)
        if abs(overlap) < 1.0e-12:
            raise FloatingPointError("transfer fixed points are nearly orthogonal.")
        right = right / overlap
        return left, right, value

    def insertion_expectation(self, operator, *, canonical: bool = True):
        """Return ``Tr[O rho O^dag]`` for a single insertion operator."""
        op = np.asarray(operator)
        if op.shape != (self.bond_dim, self.bond_dim):
            raise ValueError("operator must have shape (bond_dim, bond_dim).")
        if canonical:
            rho = self.right_fixed_density()
            return _real_if_close(np.trace(op @ rho @ op.conj().T))
        left, right, _value = self.dominant_fixed_points()
        insertion = np.kron(op, op.conj())
        return _real_if_close(np.vdot(left, insertion @ right))

    def _propagated_trace(self, initial, final, distances, *, method: str = "eig"):
        """Return ``Tr[final exp(T x)(initial)]`` for non-negative distances."""
        initial = np.asarray(initial)
        final = np.asarray(final)
        if initial.shape != (self.bond_dim, self.bond_dim):
            raise ValueError("initial must have shape (bond_dim, bond_dim).")
        if final.shape != (self.bond_dim, self.bond_dim):
            raise ValueError("final must have shape (bond_dim, bond_dim).")
        xlist, scalar = _distance_array(distances)
        transfer = self.transfer_matrix()
        vector0 = initial.reshape(-1)
        values = []
        if method == "eig":
            evals, evecs = np.linalg.eig(transfer)
            try:
                coeff = np.linalg.solve(evecs, vector0)
            except np.linalg.LinAlgError:
                method = "expm"
            else:
                for distance in xlist:
                    evolved = evecs @ (np.exp(evals * float(distance)) * coeff)
                    matrix = evolved.reshape(self.bond_dim, self.bond_dim)
                    values.append(np.trace(final @ matrix))
        if method == "expm":
            from scipy.linalg import expm

            for distance in xlist:
                evolved = expm(transfer * float(distance)) @ vector0
                matrix = evolved.reshape(self.bond_dim, self.bond_dim)
                values.append(np.trace(final @ matrix))
        elif method != "eig":
            raise ValueError("method must be 'eig' or 'expm'.")
        out = _real_if_close(np.asarray(values))
        return out[0] if scalar else out

    def two_point_correlation(
        self,
        distances,
        *,
        final_operator,
        initial_operator=None,
        initial_matrix=None,
        rho=None,
        method: str = "eig",
    ):
        r"""Generic auxiliary two-point correlator.

        This evaluates

        $$
        \mathrm{Tr}[O_f e^{x\mathcal{T}}(O_i\rho)]
        $$

        when ``initial_operator`` is supplied, or
        ``Tr[O_f exp(x T)(initial_matrix)]`` for an explicit initial matrix.
        """
        if initial_matrix is None:
            if initial_operator is None:
                raise ValueError("initial_operator or initial_matrix is required.")
            rho = self.right_fixed_density() if rho is None else np.asarray(rho)
            initial_matrix = np.asarray(initial_operator) @ rho
        return self._propagated_trace(
            initial_matrix,
            np.asarray(final_operator),
            distances,
            method=method,
        )

    def field_correlation(
        self,
        distances,
        *,
        field: int = 0,
        rho=None,
        backend: str = "transfer",
        method: str = "eig",
    ):
        """Return ``<psi_field^dag(x) psi_field(0)>`` for ``x >= 0``."""
        r = self.r_ops[int(field)]
        rho = self.right_fixed_density() if rho is None else np.asarray(rho)
        backend_key = backend.lower()
        if backend_key == "transfer":
            return self.two_point_correlation(
                distances,
                final_operator=r.conj().T,
                initial_operator=r,
                rho=rho,
                method=method,
            )
        if backend_key == "lindblad":
            solver = self.to_lindblad_solver()
            solver.eigenstates()
            values = _real_if_close(solver.correlation_2op_1t(rho, [r.conj().T, r], np.atleast_1d(distances)))
            return values[0] if np.asarray(distances).ndim == 0 else values
        raise ValueError("backend must be 'transfer' or 'lindblad'.")

    def anyonic_field_correlation(
        self,
        distances,
        *,
        statistical_angle: float,
        field: int = 0,
        density: float | None = None,
        normalized: bool = False,
        method: str = "eig",
    ):
        r"""Return the anyonic string correlator for ``x >= 0``.

        This evaluates

        $$
        \langle\psi^\dagger(x)
        \exp(i\theta\int_0^x n(y)dy)\psi(0)\rangle
        $$

        using the counting-field transfer generator

        $$
        T_\theta=T+(e^{i\theta}-1)R\otimes\bar R.
        $$

        Supplying ``density`` applies the fixed-density continuum rescaling.
        With ``normalized=True`` the result is divided by that density, so the
        zero-distance value is one.
        """
        xlist, scalar = _distance_array(distances)
        angle = float(statistical_angle)
        if not np.isfinite(angle):
            raise ValueError("statistical_angle must be finite.")
        r = np.asarray(self.r_ops[int(field)])
        dim = self.bond_dim
        eye_virtual = np.eye(dim, dtype=np.result_type(r, np.complex128))
        transfer = self.transfer_matrix()
        left, right, eigenvalue = self.dominant_fixed_points()
        density_insertion = np.kron(r, r.conj())
        raw_density = float(np.real(np.vdot(left, density_insertion @ right)))
        if raw_density <= 0.0:
            raise FloatingPointError("stationary density must be positive.")
        if density is None:
            scale = 1.0
            physical_density = raw_density
        else:
            physical_density = float(density)
            if physical_density <= 0.0:
                raise ValueError("density must be positive.")
            scale = physical_density / raw_density

        shifted = transfer - eigenvalue * np.eye(transfer.shape[0], dtype=complex)
        twisted = shifted + (np.exp(1.0j * angle) - 1.0) * density_insertion
        ket_insertion = np.kron(r, eye_virtual)
        bra_insertion = np.kron(eye_virtual, r.conj())
        initial = ket_insertion @ right
        values = []
        if method == "eig":
            eigenvalues, eigenvectors = np.linalg.eig(twisted)
            try:
                coefficients = np.linalg.solve(eigenvectors, initial)
            except np.linalg.LinAlgError:
                method = "expm"
            else:
                for distance in xlist:
                    evolved = eigenvectors @ (
                        np.exp(eigenvalues * (scale * float(distance))) * coefficients
                    )
                    values.append(scale * np.vdot(left, bra_insertion @ evolved))
        if method == "expm":
            from scipy.linalg import expm

            for distance in xlist:
                evolved = expm(twisted * (scale * float(distance))) @ initial
                values.append(scale * np.vdot(left, bra_insertion @ evolved))
        elif method != "eig":
            raise ValueError("method must be 'eig' or 'expm'.")

        result = np.asarray(values, dtype=complex)
        if normalized:
            result = result / physical_density
        return result[0] if scalar else result

    def density_correlation(
        self,
        distances,
        *,
        field: int = 0,
        rho=None,
        connected: bool = False,
        backend: str = "transfer",
        method: str = "eig",
    ):
        """Return the point-split normal density correlator.

        For ``x > 0`` this is ``<n_field(x) n_field(0)>``.  At ``x = 0`` it
        returns the normal-ordered contact value; it does not add the continuum
        delta-function singularity.
        """
        r = self.r_ops[int(field)]
        rho = self.right_fixed_density() if rho is None else np.asarray(rho)
        density_op = r.conj().T @ r
        backend_key = backend.lower()
        if backend_key == "transfer":
            values = self.two_point_correlation(
                distances,
                final_operator=density_op,
                initial_matrix=r @ rho @ r.conj().T,
                method=method,
            )
        elif backend_key == "lindblad":
            solver = self.to_lindblad_solver()
            solver.eigenstates()
            values = solver.correlation_3op_1t(
                rho,
                [r.conj().T, density_op, r],
                np.atleast_1d(distances),
            )
            values = _real_if_close(values)
        else:
            raise ValueError("backend must be 'transfer' or 'lindblad'.")
        if connected:
            density = self.insertion_expectation(r)
            values = values - density * density
        scalar = np.asarray(distances).ndim == 0
        values = _real_if_close(values)
        if backend_key == "lindblad" and scalar:
            return values[0]
        return values

    def lieb_liniger_observables(self, *, coupling: float, mu: float, canonical: bool = True):
        """Energy-density terms for the single-component Lieb-Liniger model."""
        if self.num_fields != 1:
            raise ValueError("Lieb-Liniger observables require a single field matrix R.")
        r = self.r
        commutator = self.q @ r - r @ self.q
        rr = r @ r
        density = self.insertion_expectation(r, canonical=canonical)
        kinetic = self.insertion_expectation(commutator, canonical=canonical)
        contact = self.insertion_expectation(rr, canonical=canonical)
        energy = kinetic + float(coupling) * contact - float(mu) * density
        return {
            "energy_density": float(np.real_if_close(energy)),
            "density": float(np.real_if_close(density)),
            "kinetic": float(np.real_if_close(kinetic)),
            "contact": float(np.real_if_close(contact)),
        }

    def energy_density_lieb_liniger(self, *, coupling: float, mu: float, canonical: bool = True):
        return self.lieb_liniger_observables(
            coupling=coupling,
            mu=mu,
            canonical=canonical,
        )["energy_density"]

    def lieb_liniger_fixed_density_observables(
        self,
        *,
        coupling: float,
        density: float = 1.0,
        canonical: bool = True,
    ):
        """Return Lieb-Liniger energy terms after analytic density rescaling."""
        target_density = float(density)
        if target_density <= 0.0:
            raise ValueError("density must be positive.")
        raw = self.lieb_liniger_observables(coupling=coupling, mu=0.0, canonical=canonical)
        raw_density = float(raw["density"])
        if raw_density <= 0.0:
            raise FloatingPointError("raw cMPS density must be positive for fixed-density scaling.")
        scale = target_density / raw_density
        kinetic = scale**3 * float(raw["kinetic"])
        contact = scale**2 * float(raw["contact"])
        energy = kinetic + float(coupling) * contact
        return {
            "energy_density": float(energy),
            "density": target_density,
            "kinetic": float(kinetic),
            "contact": float(contact),
            "raw_density": raw_density,
            "scale": float(scale),
        }

    def exponential_density_correlation_integral(
        self,
        decay_rate: float,
        *,
        field: int = 0,
        canonical: bool = True,
        connected: bool = False,
    ):
        r"""Return ``\int_0^\infty dr exp(-alpha r) < :n(r)n(0): >``.

        For ``canonical=False`` the dominant transfer fixed points are used,
        which is the correct path for explicit cLETTA memory states because the
        enlarged matrices are not generally left canonical.
        """
        rate = float(decay_rate)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("decay_rate must be finite and positive.")
        r = self.r_ops[int(field)]
        dim = self.bond_dim
        transfer = self.transfer_matrix()
        insertion = np.kron(r, r.conj())
        identity = np.eye(transfer.shape[0], dtype=transfer.dtype)

        if canonical:
            rho = self.right_fixed_density()
            initial = r @ rho @ r.conj().T
            solved = np.linalg.solve(transfer - rate * identity, initial.reshape(-1))
            density_op = r.conj().T @ r
            value = -np.trace(density_op @ solved.reshape(dim, dim))
            if connected:
                density = np.trace(density_op @ rho)
                value = value - density * density / rate
            return float(np.real(value))

        left, right, eigenvalue = self.dominant_fixed_points()
        shifted = transfer - eigenvalue * identity
        initial = insertion @ right
        solved = np.linalg.solve(shifted - rate * identity, initial)
        value = -np.vdot(left, insertion @ solved)
        if connected:
            density = np.vdot(left, insertion @ right)
            value = value - density * density / rate
        return float(np.real(value))

    def exponential_density_interaction(
        self,
        decay_rates,
        strengths=None,
        *,
        field: int = 0,
        canonical: bool = True,
        connected: bool = False,
    ):
        r"""Return ``\int_0^\infty dr V(r) < :n(r)n(0): >``.

        The kernel is ``V(r)=sum_i strengths[i] exp(-decay_rates[i] r)``.
        """
        rates, weights = _exponential_kernel_terms(decay_rates, strengths)
        terms = [
            weight
            * self.exponential_density_correlation_integral(
                rate,
                field=field,
                canonical=canonical,
                connected=connected,
            )
            for rate, weight in zip(rates, weights)
        ]
        return float(np.real_if_close(np.sum(terms)))

    def exponential_bose_gas_observables(
        self,
        *,
        decay_rates,
        strengths=None,
        mu: float = 0.0,
        canonical: bool = True,
        connected: bool = False,
        contact_coupling: float = 0.0,
    ):
        """Energy-density terms for a 1D Bose gas with exponential interactions."""
        if self.num_fields != 1:
            raise ValueError("exponential Bose-gas observables require a single field matrix R.")
        contact_coupling = float(contact_coupling)
        if not np.isfinite(contact_coupling):
            raise ValueError("contact_coupling must be finite.")
        r = self.r
        commutator = self.q @ r - r @ self.q
        density = self.insertion_expectation(r, canonical=canonical)
        kinetic = self.insertion_expectation(commutator, canonical=canonical)
        interaction = self.exponential_density_interaction(
            decay_rates,
            strengths,
            canonical=canonical,
            connected=connected,
        )
        contact = self.insertion_expectation(r @ r, canonical=canonical)
        energy = kinetic + interaction + contact_coupling * contact - float(mu) * density
        return {
            "energy_density": float(np.real_if_close(energy)),
            "density": float(np.real_if_close(density)),
            "kinetic": float(np.real_if_close(kinetic)),
            "contact": float(np.real_if_close(contact)),
            "interaction": float(np.real_if_close(interaction)),
        }

    def exponential_bose_gas_fixed_density_observables(
        self,
        *,
        decay_rates,
        strengths=None,
        density: float = 1.0,
        canonical: bool = True,
        connected: bool = False,
        contact_coupling: float = 0.0,
    ):
        """Return fixed-density energy terms for an exponential-kernel Bose gas."""
        target_density = float(density)
        contact_coupling = float(contact_coupling)
        if target_density <= 0.0:
            raise ValueError("density must be positive.")
        if not np.isfinite(contact_coupling):
            raise ValueError("contact_coupling must be finite.")
        rates, weights = _exponential_kernel_terms(decay_rates, strengths)
        r = self.r
        commutator = self.q @ r - r @ self.q
        raw_density = float(np.real_if_close(self.insertion_expectation(r, canonical=canonical)))
        if raw_density <= 0.0:
            raise FloatingPointError("raw cMPS density must be positive for fixed-density scaling.")
        scale = target_density / raw_density
        kinetic = scale**3 * float(
            np.real_if_close(self.insertion_expectation(commutator, canonical=canonical))
        )
        interaction_terms = [
            scale
            * weight
            * self.exponential_density_correlation_integral(
                rate / scale,
                canonical=canonical,
                connected=connected,
            )
            for rate, weight in zip(rates, weights)
        ]
        interaction = float(np.real_if_close(np.sum(interaction_terms)))
        contact = scale**2 * float(
            np.real_if_close(self.insertion_expectation(r @ r, canonical=canonical))
        )
        energy = kinetic + interaction + contact_coupling * contact
        return {
            "energy_density": float(energy),
            "density": target_density,
            "kinetic": float(kinetic),
            "contact": float(contact),
            "interaction": float(interaction),
            "raw_density": raw_density,
            "scale": float(scale),
        }

    @classmethod
    def optimize_exponential_bose_gas_fixed_density(
        cls,
        *,
        bond_dim: int,
        decay_rates,
        strengths=None,
        density: float = 1.0,
        seed_thetas=(),
        restarts: int = 8,
        seed=None,
        maxiter: int = 600,
        maxfun: int | None = None,
        method: str = "L-BFGS-B",
        regularization: float = 0.0,
        density_gauge_penalty: float = 1.0e-3,
        connected: bool = False,
        contact_coupling: float = 0.0,
        initial_scales=(0.03, 0.06, 0.1, 0.2, 0.4),
    ):
        """Optimize a continuum cMPS for a fixed-density exponential Bose gas."""
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scipy is required for cMPS optimization.") from exc

        bond_dim = int(bond_dim)
        target_density = float(density)
        if target_density <= 0.0:
            raise ValueError("density must be positive.")
        rates, weights = _exponential_kernel_terms(decay_rates, strengths)
        rng = np.random.default_rng(seed)
        candidates = [np.asarray(theta, dtype=float) for theta in seed_thetas]
        if bond_dim == 1:
            product = pack_canonical_parameters([], np.array([[np.sqrt(target_density)]]))
            candidates.insert(0, product)
        scales = tuple(float(value) for value in initial_scales) or (0.25,)
        while len(candidates) < int(restarts):
            scale = scales[len(candidates) % len(scales)]
            candidates.append(cls.random_canonical_parameters(bond_dim, rng=rng, scale=scale))

        evaluations = 0

        def objective(theta):
            nonlocal evaluations
            evaluations += 1
            theta = np.asarray(theta, dtype=float)
            try:
                state = cls.from_canonical_parameters(theta, bond_dim)
                values = state.exponential_bose_gas_fixed_density_observables(
                    decay_rates=rates,
                    strengths=weights,
                    density=target_density,
                    connected=connected,
                    contact_coupling=contact_coupling,
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError, OverflowError):
                return 1.0e30
            energy = float(values["energy_density"])
            raw_density = float(values["raw_density"])
            if not np.isfinite(energy) or raw_density <= 0.0:
                return 1.0e30
            gauge = float(density_gauge_penalty) * float(np.log(raw_density / target_density) ** 2)
            return energy + gauge + float(regularization) * float(np.dot(theta, theta))

        best = None
        for theta0 in candidates:
            options = {"maxiter": int(maxiter), "maxls": 80}
            if maxfun is not None:
                options["maxfun"] = int(maxfun)
            result = minimize(
                objective,
                np.asarray(theta0, dtype=float),
                method=method,
                options=options,
            )
            if best is None or float(result.fun) < float(best.fun):
                best = result

        state = cls.from_canonical_parameters(best.x, bond_dim)
        values = state.exponential_bose_gas_fixed_density_observables(
            decay_rates=rates,
            strengths=weights,
            density=target_density,
            connected=connected,
            contact_coupling=contact_coupling,
        )
        state.energy = values["energy_density"]
        state.density = values["density"]
        state.kinetic = values["kinetic"]
        state.contact = values["contact"]
        state.interaction = values["interaction"]
        state.raw_density = values["raw_density"]
        state.scale = values["scale"]
        state.success = bool(best.success)
        state.message = str(best.message)
        state.nfev = int(evaluations)
        state.algorithm = f"fixed-density-exponential-bose-scipy-{method}"
        return state

    @classmethod
    def optimize_exponential_bose_gas_cletta_fixed_density(
        cls,
        *,
        bond_dim: int,
        interaction_decay_rates,
        strengths=None,
        density: float = 1.0,
        num_modes: int = 1,
        depth: int = 1,
        memory_decay_rates=None,
        memory_frequencies=None,
        optimize_memory_rates: bool = True,
        optimize_memory_frequencies: bool = True,
        seed_parameters=(),
        seed_base_thetas=(),
        restarts: int = 8,
        seed=None,
        maxiter: int = 600,
        method: str = "L-BFGS-B",
        regularization: float = 1.0e-9,
        density_gauge_penalty: float = 1.0e-3,
        use_jax: bool = True,
        connected: bool = False,
        contact_coupling: float = 0.0,
        rate_bounds=None,
        frequency_bounds=None,
        initial_scales=(0.03, 0.06, 0.1, 0.2, 0.4),
        tie_scale: float = 0.05,
        rate_jitter: float = 0.35,
        frequency_jitter: float = 0.35,
        eigensolver: str = "auto",
        eigen_iterations: int = 256,
        conjugate_pair: bool = False,
        contraction_backend: str = "explicit",
        iterative_tolerance: float = 1.0e-8,
        iterative_maxiter: int | None = None,
    ):
        """Optimize a continuum cLETTA ansatz for an exponential Bose gas.

        With ``conjugate_pair=True``, exactly two memory modes are constrained
        to poles ``gamma +/- 1j*omega`` with a shared real tie matrix.  This is
        the conjugate-pair form compatible with the real canonical base
        parameterization used by this optimizer.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scipy is required for cLETTA optimization.") from exc

        bond_dim = int(bond_dim)
        num_modes = int(num_modes)
        depth = int(depth)
        target_density = float(density)
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive.")
        if num_modes < 1:
            raise ValueError("num_modes must be positive.")
        conjugate_pair = bool(conjugate_pair)
        if conjugate_pair and num_modes != 2:
            raise ValueError("conjugate_pair requires num_modes=2.")
        if depth < 0:
            raise ValueError("depth must be non-negative.")
        if target_density <= 0.0:
            raise ValueError("density must be positive.")
        contraction_backend = str(contraction_backend).lower().replace("-", "_")
        if contraction_backend not in {
            "explicit",
            "pseudomode",
            "hierarchy",
            "heom",
            "hierarchy_iterative",
            "heom_iterative",
            "matrix_free",
        }:
            raise ValueError("unsupported cLETTA contraction backend.")
        matrix_free_backend = contraction_backend in {
            "hierarchy_iterative",
            "heom_iterative",
            "matrix_free",
        }
        interaction_rates, weights = _exponential_kernel_terms(interaction_decay_rates, strengths)

        if memory_decay_rates is None:
            if interaction_rates.size == num_modes:
                reference_rates = np.array(interaction_rates, copy=True)
            else:
                rate_scale = max(target_density, float(np.mean(interaction_rates)), 1.0e-8)
                reference_rates = rate_scale * np.geomspace(0.25, 4.0, num_modes)
        else:
            reference_rates = np.atleast_1d(np.asarray(memory_decay_rates, dtype=float))
            if reference_rates.size != num_modes:
                raise ValueError("memory_decay_rates must contain num_modes values.")
            if np.any(~np.isfinite(reference_rates)) or np.any(reference_rates <= 0.0):
                raise ValueError("memory_decay_rates must be finite and positive.")
        if memory_frequencies is None:
            reference_frequencies = np.zeros(num_modes, dtype=float)
        else:
            reference_frequencies = np.atleast_1d(np.asarray(memory_frequencies, dtype=float))
            if reference_frequencies.size != num_modes:
                raise ValueError("memory_frequencies must contain num_modes values.")
            if np.any(~np.isfinite(reference_frequencies)):
                raise ValueError("memory_frequencies must be finite.")

        if conjugate_pair:
            pair_rate = float(np.sqrt(reference_rates[0] * reference_rates[1]))
            pair_frequency = 0.5 * float(reference_frequencies[0] - reference_frequencies[1])
            reference_rates = np.array([pair_rate, pair_rate], dtype=float)
            reference_frequencies = np.array([pair_frequency, -pair_frequency], dtype=float)

        if rate_bounds is None:
            lower_rate = 1.0e-4 * min(target_density, float(np.min(interaction_rates)))
            upper_rate = 1.0e3 * max(target_density, float(np.max(interaction_rates)), 1.0)
        else:
            lower_rate, upper_rate = (float(rate_bounds[0]), float(rate_bounds[1]))
            if not (0.0 < lower_rate < upper_rate):
                raise ValueError("rate_bounds must satisfy 0 < lower < upper.")
        if frequency_bounds is None:
            max_frequency = 1.0e3 * max(target_density, float(np.max(interaction_rates)), 1.0)
            lower_frequency = -max_frequency
            upper_frequency = max_frequency
        else:
            lower_frequency, upper_frequency = (float(frequency_bounds[0]), float(frequency_bounds[1]))
            if not (lower_frequency < upper_frequency):
                raise ValueError("frequency_bounds must satisfy lower < upper.")
        if conjugate_pair and not np.isclose(lower_frequency, -upper_frequency):
            raise ValueError("conjugate_pair requires frequency bounds symmetric about zero.")

        base_size = canonical_parameter_size(bond_dim)
        expanded_tie_size = num_modes * bond_dim * bond_dim
        tie_size = bond_dim * bond_dim if conjugate_pair else expanded_tie_size
        mode_parameter_count = 1 if conjugate_pair else num_modes
        full_size = (
            base_size
            + tie_size
            + (mode_parameter_count if optimize_memory_rates else 0)
            + (mode_parameter_count if optimize_memory_frequencies else 0)
        )
        expanded_size = (
            base_size
            + expanded_tie_size
            + (num_modes if optimize_memory_rates else 0)
            + (num_modes if optimize_memory_frequencies else 0)
        )
        rng = np.random.default_rng(seed)

        def pack(base_theta, ties, rates, frequencies):
            ties = np.asarray(ties, dtype=float).reshape(num_modes, bond_dim, bond_dim)
            rates = np.asarray(rates, dtype=float).reshape(num_modes)
            frequencies = np.asarray(frequencies, dtype=float).reshape(num_modes)
            if conjugate_pair:
                tie_parameters = 0.5 * (ties[0] + ties[1])
                rate_parameters = np.array([np.sqrt(rates[0] * rates[1])])
                frequency_parameters = np.array([0.5 * (frequencies[0] - frequencies[1])])
            else:
                tie_parameters = ties
                rate_parameters = rates
                frequency_parameters = frequencies
            pieces = [
                np.asarray(base_theta, dtype=float).ravel(),
                tie_parameters.reshape(-1),
            ]
            if optimize_memory_rates:
                clipped = np.clip(rate_parameters, lower_rate, upper_rate)
                pieces.append(np.log(clipped))
            if optimize_memory_frequencies:
                clipped = np.clip(frequency_parameters, lower_frequency, upper_frequency)
                pieces.append(clipped)
            return np.concatenate(pieces)

        def unpack(parameters):
            parameters = np.asarray(parameters, dtype=float)
            if parameters.size != full_size:
                raise ValueError(f"parameter size {parameters.size} does not match {full_size}.")
            base_theta = parameters[:base_size]
            offset = base_size
            tie_parameters = parameters[offset : offset + tie_size]
            offset += tie_size
            if conjugate_pair:
                tie = tie_parameters.reshape(bond_dim, bond_dim)
                ties = np.stack([tie, tie])
            else:
                ties = tie_parameters.reshape(num_modes, bond_dim, bond_dim)
            if optimize_memory_rates:
                rate_parameters = np.exp(
                    np.clip(
                        parameters[offset : offset + mode_parameter_count],
                        np.log(lower_rate),
                        np.log(upper_rate),
                    )
                )
                offset += mode_parameter_count
                rates = (
                    np.repeat(rate_parameters, 2)
                    if conjugate_pair
                    else rate_parameters
                )
            else:
                rates = reference_rates
            if optimize_memory_frequencies:
                frequency_parameters = np.clip(
                    parameters[offset : offset + mode_parameter_count],
                    lower_frequency,
                    upper_frequency,
                )
                frequencies = (
                    np.array([frequency_parameters[0], -frequency_parameters[0]])
                    if conjugate_pair
                    else frequency_parameters
                )
            else:
                frequencies = reference_frequencies
            return base_theta, ties, rates, frequencies

        def build_state(parameters):
            base_theta, ties, rates, frequencies = unpack(parameters)
            base = cls.from_canonical_parameters(base_theta, bond_dim)
            state = base.cletta_memory_state(ties, rates, depth=depth, frequencies=frequencies)
            state.cletta_parameters = np.asarray(parameters, dtype=float).copy()
            return state

        candidates = [np.asarray(theta, dtype=float) for theta in seed_parameters]
        base_candidates = [np.asarray(theta, dtype=float) for theta in seed_base_thetas]
        if bond_dim == 1:
            product = pack_canonical_parameters([], np.array([[np.sqrt(target_density)]]))
            base_candidates.insert(0, product)
        zero_ties = np.zeros((num_modes, bond_dim, bond_dim), dtype=float)
        for base_theta in base_candidates:
            if len(candidates) >= int(restarts):
                break
            candidates.append(pack(base_theta, zero_ties, reference_rates, reference_frequencies))
        random_base_candidates = list(base_candidates)
        scales = tuple(float(value) for value in initial_scales) or (0.1,)
        while len(candidates) < int(restarts):
            if random_base_candidates:
                base_theta = random_base_candidates.pop(0)
            else:
                scale = scales[len(candidates) % len(scales)]
                base_theta = cls.random_canonical_parameters(bond_dim, rng=rng, scale=scale)
            ties = float(tie_scale) * rng.normal(size=(num_modes, bond_dim, bond_dim))
            rates = reference_rates * np.exp(float(rate_jitter) * rng.normal(size=num_modes))
            frequencies = reference_frequencies + float(frequency_jitter) * rates * rng.normal(size=num_modes)
            candidates.append(pack(base_theta, ties, rates, frequencies))

        evaluations = 0

        def evaluate_observables(parameters):
            if contraction_backend in {"explicit", "pseudomode"}:
                state = build_state(parameters)
                return state.exponential_bose_gas_fixed_density_observables(
                    decay_rates=interaction_rates,
                    strengths=weights,
                    density=target_density,
                    canonical=False,
                    connected=connected,
                    contact_coupling=contact_coupling,
                )
            base_theta, ties, rates, frequencies = unpack(parameters)
            base = cls.from_canonical_parameters(base_theta, bond_dim)
            return base.cletta_exponential_bose_gas_fixed_density_observables(
                ties,
                rates,
                interaction_decay_rates=interaction_rates,
                strengths=weights,
                density=target_density,
                depth=depth,
                frequencies=frequencies,
                connected=connected,
                contact_coupling=contact_coupling,
                contraction_backend=contraction_backend,
                iterative_tolerance=iterative_tolerance,
                iterative_maxiter=iterative_maxiter,
            )

        def objective(parameters):
            nonlocal evaluations
            evaluations += 1
            parameters = np.asarray(parameters, dtype=float)
            if not np.all(np.isfinite(parameters)):
                return 1.0e30
            try:
                values = evaluate_observables(parameters)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError, OverflowError):
                return 1.0e30
            energy = float(values["energy_density"])
            raw_density = float(values["raw_density"])
            if not np.isfinite(energy) or raw_density <= 0.0:
                return 1.0e30
            gauge = float(density_gauge_penalty) * float(np.log(raw_density / target_density) ** 2)
            regularizer = float(regularization) * float(np.dot(parameters, parameters))
            return energy + gauge + regularizer

        bounds = None
        if optimize_memory_rates or optimize_memory_frequencies:
            bounds = [(None, None)] * (base_size + tie_size)
            if optimize_memory_rates:
                bounds.extend(
                    [(np.log(lower_rate), np.log(upper_rate))] * mode_parameter_count
                )
            if optimize_memory_frequencies:
                bounds.extend(
                    [(lower_frequency, upper_frequency)] * mode_parameter_count
                )

        gradient_backend = "finite-diff"
        jax_value_gradient = None
        if (
            matrix_free_backend
            and depth > 0
            and not optimize_memory_rates
            and not optimize_memory_frequencies
        ):
            try:
                jax_value_gradient = (
                    cls._exponential_bose_gas_cletta_fixed_density_sparse_implicit_value_gradient(
                        bond_dim=bond_dim,
                        depth=depth,
                        interaction_rates=interaction_rates,
                        weights=weights,
                        target_density=target_density,
                        memory_rates=reference_rates,
                        memory_frequencies=reference_frequencies,
                        contact_coupling=contact_coupling,
                        regularization=regularization,
                        density_gauge_penalty=density_gauge_penalty,
                        tolerance=iterative_tolerance,
                        maxiter=(
                            iterative_maxiter if iterative_maxiter is not None else 200
                        ),
                        connected=connected,
                        conjugate_pair=conjugate_pair,
                    )
                )
                gradient_backend = "adjoint-implicit-heom"
            except (ImportError, NotImplementedError, TypeError, ValueError):
                jax_value_gradient = None
        if use_jax and jax_value_gradient is None:
            try:
                expanded_value_gradient = cls._exponential_bose_gas_cletta_fixed_density_jax_value_gradient(
                    bond_dim=bond_dim,
                    num_modes=num_modes,
                    depth=depth,
                    base_size=base_size,
                    tie_size=expanded_tie_size,
                    interaction_rates=interaction_rates,
                    weights=weights,
                    target_density=target_density,
                    reference_rates=reference_rates,
                    reference_frequencies=reference_frequencies,
                    optimize_memory_rates=optimize_memory_rates,
                    optimize_memory_frequencies=optimize_memory_frequencies,
                    lower_rate=lower_rate,
                    upper_rate=upper_rate,
                    lower_frequency=lower_frequency,
                    upper_frequency=upper_frequency,
                    connected=connected,
                    contact_coupling=contact_coupling,
                    regularization=0.0 if conjugate_pair else regularization,
                    density_gauge_penalty=density_gauge_penalty,
                    eigensolver=eigensolver,
                    eigen_iterations=eigen_iterations,
                    linear_tolerance=iterative_tolerance,
                    linear_maxiter=(
                        iterative_maxiter if iterative_maxiter is not None else 200
                    ),
                )
                if conjugate_pair:
                    def expand_pair_parameters(parameters):
                        parameters = np.asarray(parameters, dtype=float)
                        base = parameters[:base_size]
                        offset = base_size
                        tie = parameters[offset : offset + tie_size]
                        offset += tie_size
                        pieces = [base, tie, tie]
                        if optimize_memory_rates:
                            log_rate = parameters[offset : offset + 1]
                            offset += 1
                            pieces.extend([log_rate, log_rate])
                        if optimize_memory_frequencies:
                            frequency = parameters[offset : offset + 1]
                            pieces.extend([frequency, -frequency])
                        expanded = np.concatenate(pieces)
                        if expanded.size != expanded_size:
                            raise ValueError(
                                f"expanded parameter size {expanded.size} does not match {expanded_size}."
                            )
                        return expanded

                    def pull_back_pair_gradient(gradient):
                        gradient = np.asarray(gradient, dtype=float)
                        reduced = np.zeros(full_size, dtype=float)
                        reduced[:base_size] = gradient[:base_size]
                        full_offset = base_size
                        reduced_offset = base_size
                        one_tie_size = bond_dim * bond_dim
                        reduced[reduced_offset : reduced_offset + one_tie_size] = (
                            gradient[full_offset : full_offset + one_tie_size]
                            + gradient[
                                full_offset + one_tie_size : full_offset + 2 * one_tie_size
                            ]
                        )
                        full_offset += 2 * one_tie_size
                        reduced_offset += one_tie_size
                        if optimize_memory_rates:
                            reduced[reduced_offset] = gradient[full_offset] + gradient[full_offset + 1]
                            full_offset += 2
                            reduced_offset += 1
                        if optimize_memory_frequencies:
                            reduced[reduced_offset] = gradient[full_offset] - gradient[full_offset + 1]
                        return reduced

                    def pair_value_gradient(parameters):
                        parameters = np.asarray(parameters, dtype=float)
                        value, expanded_gradient = expanded_value_gradient(
                            expand_pair_parameters(parameters)
                        )
                        gradient = pull_back_pair_gradient(expanded_gradient)
                        if regularization:
                            value += float(regularization) * float(np.dot(parameters, parameters))
                            gradient += 2.0 * float(regularization) * parameters
                        return value, gradient

                    jax_value_gradient = pair_value_gradient
                else:
                    jax_value_gradient = expanded_value_gradient
                gradient_backend = "jax-matrix-free" if matrix_free_backend else "jax"
            except (ImportError, NotImplementedError, TypeError, ValueError):
                jax_value_gradient = None

        results = []
        for theta0 in candidates:
            if jax_value_gradient is None:
                result = minimize(
                    objective,
                    np.asarray(theta0, dtype=float),
                    method=method,
                    bounds=bounds,
                    options={"maxiter": int(maxiter), "maxls": 80},
                )
            else:
                cache = {"theta": None, "value": None, "gradient": None}

                def cached(theta):
                    nonlocal evaluations
                    theta_array = np.asarray(theta, dtype=float)
                    if cache["theta"] is not None and np.array_equal(theta_array, cache["theta"]):
                        return cache["value"], cache["gradient"]
                    evaluations += 1
                    try:
                        value, gradient = jax_value_gradient(theta_array)
                    except (
                        FloatingPointError,
                        np.linalg.LinAlgError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        value = 1.0e30
                        gradient = np.zeros_like(theta_array)
                    cache["theta"] = theta_array.copy()
                    cache["value"] = value
                    cache["gradient"] = gradient
                    return value, gradient

                result = minimize(
                    lambda theta: cached(theta)[0],
                    np.asarray(theta0, dtype=float),
                    jac=lambda theta: cached(theta)[1],
                    method=method,
                    bounds=bounds,
                    options={"maxiter": int(maxiter), "maxls": 80},
                )
            results.append(result)

        best_parameters = None
        best_values = None
        best_energy = np.inf
        best_success = False
        best_message = "no valid cLETTA candidate"
        final_candidates = [
            (np.asarray(result.x, dtype=float), bool(result.success), str(result.message))
            for result in results
        ]
        final_candidates.extend(
            (
                np.asarray(theta0, dtype=float),
                False,
                "selected unoptimized cMPS/cLETTA seed",
            )
            for theta0 in candidates
        )
        for parameters, success, message in final_candidates:
            try:
                values = evaluate_observables(parameters)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError, OverflowError):
                continue
            energy = float(values["energy_density"])
            raw_density = float(values["raw_density"])
            if not np.isfinite(energy) or raw_density <= 0.0:
                continue
            if energy < best_energy:
                best_parameters = parameters
                best_values = values
                best_energy = energy
                best_success = success
                best_message = message
        if best_parameters is None or best_values is None:
            raise FloatingPointError("no valid cLETTA candidate found.")

        state = build_state(best_parameters)
        values = best_values
        state.energy = values["energy_density"]
        state.density = values["density"]
        state.kinetic = values["kinetic"]
        state.contact = values["contact"]
        state.interaction = values["interaction"]
        state.raw_density = values["raw_density"]
        state.scale = values["scale"]
        state.success = bool(best_success)
        state.message = str(best_message)
        state.nfev = int(evaluations)
        pair_label = "-conjugate-pair" if conjugate_pair else ""
        state.algorithm = (
            f"fixed-density-exponential-bose-cletta{pair_label}-{contraction_backend}-"
            f"{gradient_backend}-{method}"
        )
        return state

    @staticmethod
    def _exponential_bose_gas_cletta_fixed_density_sparse_implicit_value_gradient(
        *,
        bond_dim: int,
        depth: int,
        interaction_rates,
        weights,
        target_density: float,
        memory_rates,
        memory_frequencies,
        contact_coupling: float,
        regularization: float,
        density_gauge_penalty: float,
        tolerance: float,
        maxiter: int,
        connected: bool = False,
        conjugate_pair: bool = False,
    ):
        """Build an arbitrary-mode fixed-pole gradient using sparse implicit solves."""
        from scipy.sparse import bmat, csc_matrix, eye as sparse_eye
        from scipy.sparse.linalg import (
            LinearOperator,
            gmres,
            spilu,
            splu,
        )

        from .cletta import (
            _multimode_memory_operators,
            cletta_multimode_hierarchy_sparse_generator,
            cletta_multimode_memory_matrices,
            hierarchy_blocks_to_matrix,
            matrix_to_hierarchy_blocks,
        )

        dim = int(bond_dim)
        depth = int(depth)
        base_size = canonical_parameter_size(dim)
        pairs = skew_pairs(dim)
        interaction_rates = np.asarray(interaction_rates, dtype=float)
        weights = np.asarray(weights, dtype=float)
        memory_rates = np.asarray(memory_rates, dtype=float)
        memory_frequencies = np.asarray(memory_frequencies, dtype=float)
        target_density = float(target_density)
        contact_coupling = float(contact_coupling)
        regularization = float(regularization)
        density_gauge_penalty = float(density_gauge_penalty)
        tolerance = float(tolerance)
        maxiter = int(maxiter)
        connected = bool(connected)
        conjugate_pair = bool(conjugate_pair)
        num_modes = int(memory_rates.size)
        if num_modes < 1 or memory_frequencies.size != num_modes:
            raise ValueError("memory rates and frequencies must have the same positive size.")
        if conjugate_pair and num_modes != 2:
            raise ValueError("conjugate_pair requires exactly two memory modes.")
        tie_parameter_count = 1 if conjugate_pair else num_modes
        parameter_size = base_size + tie_parameter_count * dim * dim
        keys, _key_to_index, annihilation, _number = _multimode_memory_operators(
            num_modes,
            depth,
            np.complex128,
        )
        memory_dim = len(keys)
        block_shape = (memory_dim, memory_dim, dim, dim)
        size = int(np.prod(block_shape))
        identity_size = sparse_eye(size, dtype=np.complex128, format="csc")
        zero_scalar = csc_matrix((1, 1), dtype=np.complex128)

        def vector_to_matrix(vector):
            return hierarchy_blocks_to_matrix(np.asarray(vector).reshape(block_shape))

        def matrix_to_vector(matrix):
            return matrix_to_hierarchy_blocks(
                matrix,
                bond_dim=dim,
                memory_dim=memory_dim,
            ).reshape(-1)

        def insertion_action(vector, operator):
            matrix = vector_to_matrix(vector)
            return matrix_to_vector(operator @ matrix @ operator.conj().T)

        def insertion_adjoint_action(vector, operator):
            matrix = vector_to_matrix(vector)
            return matrix_to_vector(operator.conj().T @ matrix @ operator)

        def insertion_derivative_action(vector, operator, derivative):
            matrix = vector_to_matrix(vector)
            out = (
                derivative @ matrix @ operator.conj().T
                + operator @ matrix @ derivative.conj().T
            )
            return matrix_to_vector(out)

        def transfer_derivative_action(vector, dq, dr, q_memory, r_memory):
            matrix = vector_to_matrix(vector)
            out = dq @ matrix + matrix @ dq.conj().T
            out += dr @ matrix @ r_memory.conj().T
            out += r_memory @ matrix @ dr.conj().T
            return matrix_to_vector(out)

        def transfer_adjoint_derivative_action(vector, dq, dr, q_memory, r_memory):
            matrix = vector_to_matrix(vector)
            out = dq.conj().T @ matrix + matrix @ dq
            out += dr.conj().T @ matrix @ r_memory
            out += r_memory.conj().T @ matrix @ dr
            return matrix_to_vector(out)

        def parameter_derivative(index, base, q_memory):
            dq_base = np.zeros((dim, dim), dtype=np.complex128)
            dr_base = np.zeros((dim, dim), dtype=np.complex128)
            tie_derivatives = np.zeros(
                (num_modes, dim, dim), dtype=np.complex128
            )
            if index < len(pairs):
                row, column = pairs[index]
                dq_base[row, column] = 1.0
                dq_base[column, row] = -1.0
            elif index < base_size:
                r_index = index - len(pairs)
                dr_base.reshape(-1)[r_index] = 1.0
                dq_base = -0.5 * (
                    dr_base.T @ base.r + base.r.T @ dr_base
                )
            else:
                tie_index = index - base_size
                if conjugate_pair:
                    row, column = divmod(tie_index, dim)
                    tie_derivatives[:, row, column] = 1.0
                else:
                    mode, entry = divmod(tie_index, dim * dim)
                    row, column = divmod(entry, dim)
                    tie_derivatives[mode, row, column] = 1.0
            dq_memory = np.kron(np.eye(memory_dim), dq_base)
            dr_memory = np.kron(np.eye(memory_dim), dr_base)
            for mode in range(num_modes):
                dr_memory += np.kron(
                    annihilation[mode].conj().T,
                    tie_derivatives[mode],
                )
            return dq_memory, dr_memory

        def value_gradient(parameters):
            parameters = np.asarray(parameters, dtype=float)
            if parameters.size != parameter_size:
                raise ValueError(
                    f"implicit-gradient parameter size {parameters.size} does not match "
                    f"{parameter_size}."
                )
            base = ContinuousMPS.from_canonical_parameters(parameters[:base_size], dim)
            if conjugate_pair:
                tie = parameters[base_size:].reshape(dim, dim)
                ties = np.stack([tie, tie])
            else:
                ties = parameters[base_size:].reshape(num_modes, dim, dim)
            q_memory, r_memory = cletta_multimode_memory_matrices(
                base.q,
                base.r,
                ties,
                memory_rates,
                depth=depth,
                frequencies=memory_frequencies,
            )
            generator = cletta_multimode_hierarchy_sparse_generator(
                base.q,
                base.r,
                ties,
                memory_rates,
                depth=depth,
                frequencies=memory_frequencies,
            ).tocsc()
            initial = np.zeros(block_shape, dtype=np.complex128)
            initial[0, 0] = np.eye(dim, dtype=np.complex128)
            initial = initial.reshape(-1)
            if size > 1:
                probe = np.arange(1, size + 1, dtype=float)
                initial = initial + 1.0e-6 * probe / np.linalg.norm(probe)
            eigenvalue, left, right = _dominant_sparse_biorthogonal_pair(
                generator,
                initial,
                tolerance=tolerance,
                maxiter=maxiter,
                label="implicit-gradient cLETTA",
            )

            density_right = insertion_action(right, r_memory)
            raw_density_value = np.vdot(left, density_right)
            raw_density = float(np.real(raw_density_value))
            if raw_density <= 0.0:
                raise FloatingPointError("implicit-gradient density must be positive.")
            scale = target_density / raw_density
            commutator = q_memory @ r_memory - r_memory @ q_memory
            pair_operator = r_memory @ r_memory
            kinetic_raw = float(
                np.real(np.vdot(left, insertion_action(right, commutator)))
            )
            contact_raw = float(
                np.real(np.vdot(left, insertion_action(right, pair_operator)))
            )

            shifted_eigen = generator - eigenvalue * identity_size
            left_shifted = generator.conj().T - eigenvalue.conjugate() * identity_size
            iterative_solves = size > 2000

            eye_virtual = np.eye(dim, dtype=np.complex128)
            base_transfer = (
                np.kron(base.q, eye_virtual)
                + np.kron(eye_virtual, base.q.conj())
                + np.kron(base.r, base.r.conj())
            )
            decay_ket = np.asarray(keys, dtype=float) @ (
                memory_rates + 1.0j * memory_frequencies
            )
            decay_bra = np.asarray(keys, dtype=float) @ (
                memory_rates - 1.0j * memory_frequencies
            )
            block_size = dim * dim

            def hierarchy_block_preconditioner(alpha):
                shifts = (
                    decay_ket[:, None]
                    + decay_bra[None, :]
                    + eigenvalue
                    + float(alpha)
                )
                blocks = (
                    base_transfer[None, None, :, :]
                    - shifts[:, :, None, None]
                    * np.eye(block_size, dtype=np.complex128)[None, None, :, :]
                )
                inverses = np.linalg.inv(blocks)
                inverse_adjoints = np.swapaxes(inverses.conj(), -1, -2)

                def apply(vector):
                    shaped = np.asarray(vector).reshape(
                        memory_dim,
                        memory_dim,
                        block_size,
                    )
                    return np.einsum(
                        "ijab,ijb->ija",
                        inverses,
                        shaped,
                        optimize=True,
                    ).reshape(-1)

                def apply_adjoint(vector):
                    shaped = np.asarray(vector).reshape(
                        memory_dim,
                        memory_dim,
                        block_size,
                    )
                    return np.einsum(
                        "ijab,ijb->ija",
                        inverse_adjoints,
                        shaped,
                        optimize=True,
                    ).reshape(-1)

                return LinearOperator(
                    (size, size),
                    matvec=apply,
                    rmatvec=apply_adjoint,
                    dtype=np.complex128,
                )

            def krylov_solver(
                matrix,
                *,
                rank_right=None,
                rank_left=None,
                preconditioner=None,
            ):
                if preconditioner is None:
                    stabilization = 1.0e-4 if rank_right is not None else 0.0
                    preconditioner_matrix = matrix + stabilization * identity_size
                    preconditioner_factor = spilu(
                        preconditioner_matrix.tocsc(),
                        drop_tol=1.0e-6,
                        fill_factor=20.0,
                    )

                    preconditioner = LinearOperator(
                        matrix.shape,
                        matvec=preconditioner_factor.solve,
                        rmatvec=lambda vector: preconditioner_factor.solve(
                            vector,
                            trans="H",
                        ),
                        dtype=np.complex128,
                    )

                def action(vector):
                    out = matrix @ vector
                    if rank_right is not None:
                        out = out + rank_right * np.vdot(rank_left, vector)
                    return out

                operator = LinearOperator(
                    matrix.shape,
                    matvec=action,
                    dtype=np.complex128,
                )

                def solve(rhs):
                    solution, info = gmres(
                        operator,
                        rhs,
                        M=preconditioner,
                        rtol=max(tolerance, 1.0e-9),
                        atol=0.0,
                        restart=min(160, size),
                        maxiter=maxiter,
                    )
                    if info != 0:
                        raise FloatingPointError(
                            f"implicit-gradient GMRES did not converge (info={info})."
                        )
                    return solution

                return solve

            if iterative_solves:
                eigen_preconditioner = hierarchy_block_preconditioner(1.0e-4)
                left_adjoint_solve = krylov_solver(
                    shifted_eigen,
                    rank_right=right,
                    rank_left=left,
                    preconditioner=eigen_preconditioner,
                )
                right_adjoint_solve = krylov_solver(
                    left_shifted,
                    rank_right=left,
                    rank_left=right,
                    preconditioner=eigen_preconditioner.H,
                )
            else:
                right_border = bmat(
                    [
                        [shifted_eigen, -csc_matrix(right[:, None])],
                        [csc_matrix(left.conj()[None, :]), zero_scalar],
                    ],
                    format="csc",
                )
                left_border = bmat(
                    [
                        [left_shifted, -csc_matrix(left[:, None])],
                        [csc_matrix(right.conj()[None, :]), zero_scalar],
                    ],
                    format="csc",
                )
                right_factor = splu(right_border)
                left_factor = splu(left_border)

            resolvents = []
            interaction_raw = []
            for rate in interaction_rates:
                alpha = float(rate) / scale
                resolvent_matrix = generator - (eigenvalue + alpha) * identity_size
                if iterative_solves:
                    resolvent_preconditioner = hierarchy_block_preconditioner(alpha)
                    solve_resolvent = krylov_solver(
                        resolvent_matrix,
                        preconditioner=resolvent_preconditioner,
                    )
                    solve_resolvent_adjoint = krylov_solver(
                        resolvent_matrix.conj().T,
                        preconditioner=resolvent_preconditioner.H,
                    )
                else:
                    resolvent_factor = splu(resolvent_matrix)
                    solve_resolvent = resolvent_factor.solve

                    def solve_resolvent_adjoint(rhs, factor=resolvent_factor):
                        return factor.solve(rhs, trans="H")

                solved = solve_resolvent(density_right)
                integral = -np.vdot(left, insertion_action(solved, r_memory))
                if connected:
                    integral -= raw_density_value * raw_density_value / alpha
                resolvents.append(
                    (alpha, solve_resolvent_adjoint, solved)
                )
                interaction_raw.append(float(np.real(integral)))

            kinetic = scale**3 * kinetic_raw
            contact = scale**2 * contact_raw
            interaction = scale * float(np.dot(weights, interaction_raw))
            energy = kinetic + interaction + contact_coupling * contact
            gauge = density_gauge_penalty * np.log(raw_density / target_density) ** 2
            value = energy + gauge + regularization * float(np.dot(parameters, parameters))
            kinetic_right = insertion_action(right, commutator)
            kinetic_left = insertion_adjoint_action(left, commutator)
            contact_right = insertion_action(right, pair_operator)
            contact_left = insertion_adjoint_action(left, pair_operator)
            density_left = insertion_adjoint_action(left, r_memory)

            gradient_right = 0.5 * scale**3 * kinetic_left
            gradient_left = 0.5 * scale**3 * kinetic_right
            gradient_right += 0.5 * contact_coupling * scale**2 * contact_left
            gradient_left += 0.5 * contact_coupling * scale**2 * contact_right

            scale_derivative = 3.0 * scale**2 * kinetic_raw
            scale_derivative += 2.0 * contact_coupling * scale * contact_raw
            scale_derivative += float(np.dot(weights, interaction_raw))
            if connected:
                scale_derivative -= sum(
                    float(weight)
                    * float(np.real(raw_density_value * raw_density_value / alpha))
                    for weight, (alpha, _solve_adjoint, _solved) in zip(
                        weights, resolvents
                    )
                )
            density_coefficient = (
                -scale * scale_derivative / raw_density
                + 2.0
                * density_gauge_penalty
                * np.log(raw_density / target_density)
                / raw_density
            )
            eigenvalue_coefficient = 0.0j
            adjoint_resolvents = []
            for weight, (alpha, solve_adjoint, solved) in zip(
                weights,
                resolvents,
            ):
                weight = float(weight)
                density_solved = insertion_action(solved, r_memory)
                gradient_left -= 0.5 * scale * weight * density_solved
                gradient_solved = (
                    -0.5 * scale * weight * density_left
                )
                resolvent_adjoint = solve_adjoint(gradient_solved)
                gradient_right += insertion_adjoint_action(
                    resolvent_adjoint,
                    r_memory,
                )
                overlap_resolvent = np.vdot(resolvent_adjoint, solved)
                eigenvalue_coefficient += overlap_resolvent
                density_coefficient += (
                    2.0
                    * float(np.real(overlap_resolvent))
                    * alpha
                    / raw_density
                )
                if connected:
                    density_coefficient += (
                        -2.0
                        * scale
                        * weight
                        * raw_density_value
                        / alpha
                    )
                adjoint_resolvents.append(
                    (weight, solved, resolvent_adjoint)
                )

            gradient_left += 0.5 * density_coefficient * density_right
            gradient_right += (
                0.5 * density_coefficient.conjugate() * density_left
            )

            if iterative_solves:
                right_adjoint = right_adjoint_solve(
                    gradient_right
                    - left * np.vdot(right, gradient_right)
                )
                left_adjoint = left_adjoint_solve(
                    gradient_left
                    - right * np.vdot(left, gradient_left)
                )
            else:
                right_rhs = np.concatenate(
                    [gradient_right, np.zeros(1, dtype=np.complex128)]
                )
                left_rhs = np.concatenate(
                    [gradient_left, np.zeros(1, dtype=np.complex128)]
                )
                right_adjoint = right_factor.solve(
                    right_rhs,
                    trans="H",
                )[:size]
                left_adjoint = left_factor.solve(
                    left_rhs,
                    trans="H",
                )[:size]

            gradient = np.zeros_like(parameters)
            for parameter_index in range(parameter_size):
                dq, dr = parameter_derivative(parameter_index, base, q_memory)
                dg_right = transfer_derivative_action(
                    right,
                    dq,
                    dr,
                    q_memory,
                    r_memory,
                )
                dg_adjoint_left = transfer_adjoint_derivative_action(
                    left,
                    dq,
                    dr,
                    q_memory,
                    r_memory,
                )
                density_derivative = insertion_derivative_action(
                    right,
                    r_memory,
                    dr,
                )
                dcommutator = (
                    dq @ r_memory
                    + q_memory @ dr
                    - dr @ q_memory
                    - r_memory @ dq
                )
                kinetic_derivative = insertion_derivative_action(
                    right,
                    commutator,
                    dcommutator,
                )
                dpair = dr @ r_memory + r_memory @ dr
                contact_derivative = insertion_derivative_action(
                    right,
                    pair_operator,
                    dpair,
                )

                derivative = scale**3 * float(
                    np.real(np.vdot(left, kinetic_derivative))
                )
                derivative += contact_coupling * scale**2 * float(
                    np.real(np.vdot(left, contact_derivative))
                )
                derivative += float(
                    np.real(
                        density_coefficient
                        * np.vdot(left, density_derivative)
                    )
                )

                for weight, solved, resolvent_adjoint in adjoint_resolvents:
                    solved_density_derivative = insertion_derivative_action(
                        solved,
                        r_memory,
                        dr,
                    )
                    derivative -= scale * weight * float(
                        np.real(np.vdot(left, solved_density_derivative))
                    )
                    derivative += 2.0 * float(
                        np.real(np.vdot(resolvent_adjoint, density_derivative))
                    )
                    derivative -= 2.0 * float(
                        np.real(
                            np.vdot(
                                resolvent_adjoint,
                                transfer_derivative_action(
                                    solved,
                                    dq,
                                    dr,
                                    q_memory,
                                    r_memory,
                                ),
                            )
                        )
                    )

                derivative -= 2.0 * float(
                    np.real(np.vdot(right_adjoint, dg_right))
                )
                derivative -= 2.0 * float(
                    np.real(np.vdot(left_adjoint, dg_adjoint_left))
                )
                derivative += 2.0 * float(
                    np.real(
                        eigenvalue_coefficient
                        * np.vdot(left, dg_right)
                    )
                )
                derivative += 2.0 * regularization * parameters[parameter_index]
                gradient[parameter_index] = derivative
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                return 1.0e30, np.zeros_like(parameters)
            return float(value), gradient

        return value_gradient

    @staticmethod
    def _exponential_bose_gas_cletta_fixed_density_jax_value_gradient(
        *,
        bond_dim: int,
        num_modes: int,
        depth: int,
        base_size: int,
        tie_size: int,
        interaction_rates,
        weights,
        target_density: float,
        reference_rates,
        reference_frequencies,
        optimize_memory_rates: bool,
        optimize_memory_frequencies: bool,
        lower_rate: float,
        upper_rate: float,
        lower_frequency: float,
        upper_frequency: float,
        connected: bool,
        contact_coupling: float,
        regularization: float,
        density_gauge_penalty: float,
        eigensolver: str,
        eigen_iterations: int,
        linear_tolerance: float,
        linear_maxiter: int,
    ):
        try:
            import jax
            import jax.numpy as jnp
            from jax.scipy.sparse.linalg import gmres
        except ImportError as exc:  # pragma: no cover
            raise ImportError("jax is not available.") from exc

        from .cletta import _multimode_memory_operators

        jax.config.update("jax_enable_x64", True)
        dim = int(bond_dim)
        modes = int(num_modes)
        pairs = skew_pairs(dim)
        skew_basis = np.zeros((len(pairs), dim, dim), dtype=float)
        for index, (row, col) in enumerate(pairs):
            skew_basis[index, row, col] = 1.0
            skew_basis[index, col, row] = -1.0
        _keys, _key_to_index, annihilation, number = _multimode_memory_operators(
            modes,
            int(depth),
            np.complex128,
        )
        memory_dim = int(annihilation.shape[1])
        effective_dim = dim * memory_dim
        transfer_size = effective_dim * effective_dim
        eigensolver = str(eigensolver).lower()
        if eigensolver not in {"auto", "dense", "iterative"}:
            raise ValueError("eigensolver must be 'auto', 'dense', or 'iterative'.")
        if eigensolver == "auto":
            eigensolver = "iterative" if transfer_size > 256 else "dense"
        eigen_iterations = int(eigen_iterations)
        linear_maxiter = int(linear_maxiter)
        linear_tolerance = float(linear_tolerance)
        if eigen_iterations < 1 or linear_maxiter < 1:
            raise ValueError("eigen_iterations and linear_maxiter must be positive.")
        if not np.isfinite(linear_tolerance) or linear_tolerance <= 0.0:
            raise ValueError("linear_tolerance must be finite and positive.")

        skew_basis = jnp.asarray(skew_basis, dtype=jnp.float64)
        annihilation = jnp.asarray(annihilation, dtype=jnp.complex128)
        number = jnp.asarray(number, dtype=jnp.complex128)
        eye_virtual = jnp.eye(dim, dtype=jnp.complex128)
        eye_memory = jnp.eye(memory_dim, dtype=jnp.complex128)
        eye_effective = jnp.eye(effective_dim, dtype=jnp.complex128)
        eye_transfer = jnp.eye(transfer_size, dtype=jnp.complex128)
        trace_row_effective = eye_effective.reshape(-1)
        fixed_point_rhs = jnp.zeros((effective_dim * effective_dim,), dtype=jnp.complex128).at[0].set(1.0)
        interaction_rates = jnp.asarray(interaction_rates, dtype=jnp.float64)
        weights = jnp.asarray(weights, dtype=jnp.float64)
        reference_rates = jnp.asarray(reference_rates, dtype=jnp.float64)
        reference_frequencies = jnp.asarray(reference_frequencies, dtype=jnp.float64)
        log_lower_rate = float(np.log(lower_rate))
        log_upper_rate = float(np.log(upper_rate))
        lower_frequency = float(lower_frequency)
        upper_frequency = float(upper_frequency)
        target_density = float(target_density)
        contact_coupling = float(contact_coupling)
        regularization = float(regularization)
        density_gauge_penalty = float(density_gauge_penalty)

        def unpack(parameters):
            base_theta = parameters[:base_size]
            offset = base_size
            ties = parameters[offset : offset + tie_size].reshape((modes, dim, dim))
            offset += tie_size
            if optimize_memory_rates:
                rates = jnp.exp(
                    jnp.clip(
                        parameters[offset : offset + modes],
                        log_lower_rate,
                        log_upper_rate,
                    )
                )
                offset += modes
            else:
                rates = reference_rates
            if optimize_memory_frequencies:
                frequencies = jnp.clip(
                    parameters[offset : offset + modes],
                    lower_frequency,
                    upper_frequency,
                )
            else:
                frequencies = reference_frequencies
            return base_theta, ties, rates, frequencies

        def base_matrices(base_theta):
            skew = jnp.tensordot(base_theta[: len(pairs)], skew_basis, axes=(0, 0))
            r = base_theta[len(pairs) :].reshape((dim, dim))
            q = skew - 0.5 * (r.T @ r)
            return q.astype(jnp.complex128), r.astype(jnp.complex128)

        def memory_state(parameters):
            base_theta, ties, rates, frequencies = unpack(parameters)
            q, r = base_matrices(base_theta)
            q_memory = jnp.kron(eye_memory, q)
            r_memory = jnp.kron(eye_memory, r)
            ties = ties.astype(jnp.complex128)
            for mode in range(modes):
                q_memory = q_memory - (rates[mode] + 1.0j * frequencies[mode]) * jnp.kron(
                    number[mode],
                    eye_virtual,
                )
                r_memory = r_memory + jnp.sqrt(rates[mode]) * jnp.kron(
                    annihilation[mode],
                    eye_virtual,
                )
                r_memory = r_memory + jnp.kron(jnp.conj(annihilation[mode].T), ties[mode])
            return q_memory, r_memory, rates, frequencies

        def transfer_matvec(matrix, q, r):
            return (q @ matrix + matrix @ jnp.conj(q.T) + r @ matrix @ jnp.conj(r.T)).reshape(-1)

        def transfer_adjoint_matvec(matrix, q, r):
            return (jnp.conj(q.T) @ matrix + matrix @ q + jnp.conj(r.T) @ matrix @ r).reshape(-1)

        def dominant_fixed_points(transfer, q, r, rates, frequencies):
            if eigensolver == "dense":
                values = jnp.linalg.eigvals(transfer)
                index = jnp.argmax(jnp.real(values))
                value = values[index]

                right_matrix = (transfer - value * eye_transfer).at[0, :].set(trace_row_effective)
                right = jnp.linalg.solve(right_matrix, fixed_point_rhs)

                left_generator = jnp.conj(transfer.T) - jnp.conj(value) * eye_transfer
                left_matrix = left_generator.at[0, :].set(trace_row_effective)
                left = jnp.linalg.solve(left_matrix, fixed_point_rhs)

                overlap = jnp.vdot(left, right)
                right = right / overlap
                return left, right, value, overlap

            from pyqed.jax_eigs import dominant_eig

            right0 = trace_row_effective
            left0 = trace_row_effective
            spectral_bound = 2.0 * jnp.linalg.norm(q, ord=2) + jnp.linalg.norm(r, ord=2) ** 2
            shift = jax.lax.stop_gradient(1.0 + 0.6 * spectral_bound)
            value, left, right = dominant_eig(
                lambda vector: transfer_matvec(vector.reshape((effective_dim, effective_dim)), q, r),
                lambda vector: transfer_adjoint_matvec(vector.reshape((effective_dim, effective_dim)), q, r),
                right0,
                left0,
                iterations=eigen_iterations,
                shift=shift,
            )
            overlap = jnp.vdot(left, right)
            return left, right, value, overlap

        def insertion_expectation(left, right, operator):
            matrix = right.reshape((effective_dim, effective_dim))
            inserted = operator @ matrix @ jnp.conj(operator.T)
            return jnp.vdot(left, inserted.reshape(-1))

        def insertion_apply(vector, operator):
            matrix = vector.reshape((effective_dim, effective_dim))
            return (operator @ matrix @ jnp.conj(operator.T)).reshape(-1)

        def shifted_solve(rhs, alpha, eigenvalue, q, r, transfer):
            if eigensolver == "dense":
                shifted = transfer - (eigenvalue + alpha) * eye_transfer
                return jnp.linalg.solve(shifted, rhs)

            def action(vector):
                matrix = vector.reshape((effective_dim, effective_dim))
                return transfer_matvec(matrix, q, r) - (eigenvalue + alpha) * vector

            solution, _info = gmres(
                action,
                rhs,
                tol=linear_tolerance,
                atol=linear_tolerance,
                restart=min(40, transfer_size),
                maxiter=linear_maxiter,
                solve_method="incremental",
            )
            return solution

        @jax.jit
        def objective(parameters):
            q, r, rates, frequencies = memory_state(parameters)
            transfer = None
            if eigensolver == "dense":
                transfer = (
                    jnp.kron(q, eye_effective)
                    + jnp.kron(eye_effective, jnp.conj(q))
                    + jnp.kron(r, jnp.conj(r))
                )
            left, right, eigenvalue, overlap = dominant_fixed_points(
                transfer, q, r, rates, frequencies
            )
            density_right = insertion_apply(right, r)
            raw_density_value = jnp.vdot(left, density_right)
            raw_density = jnp.real(raw_density_value)
            safe_density = jnp.maximum(raw_density, 1.0e-12)
            scale = target_density / safe_density

            commutator = q @ r - r @ q
            kinetic = scale**3 * jnp.real(insertion_expectation(left, right, commutator))
            pair = r @ r
            contact = scale**2 * jnp.real(insertion_expectation(left, right, pair))
            interaction = jnp.array(0.0, dtype=jnp.float64)
            for rate, weight in zip(interaction_rates, weights):
                alpha = rate / scale
                solved = shifted_solve(density_right, alpha, eigenvalue, q, r, transfer)
                integral = -jnp.vdot(left, insertion_apply(solved, r))
                if connected:
                    integral = integral - raw_density_value * raw_density_value / alpha
                interaction = interaction + scale * weight * jnp.real(integral)
            energy = kinetic + interaction + contact_coupling * contact
            gauge = density_gauge_penalty * jnp.log(safe_density / target_density) ** 2
            regularizer = regularization * jnp.dot(parameters, parameters)
            bad = jnp.logical_or(raw_density <= 1.0e-12, jnp.abs(overlap) <= 1.0e-12)
            bad = jnp.logical_or(bad, jnp.logical_not(jnp.isfinite(energy)))
            return jnp.where(bad, 1.0e30, energy + gauge + regularizer)

        value_and_grad = jax.jit(jax.value_and_grad(objective))

        def value_gradient(parameters):
            parameters = np.asarray(parameters, dtype=float)
            value, gradient = value_and_grad(jnp.asarray(parameters, dtype=jnp.float64))
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                return 1.0e30, np.zeros_like(parameters)
            return value, gradient

        return value_gradient

    @classmethod
    def optimize_lieb_liniger_cletta_fixed_density(
        cls,
        *,
        bond_dim: int,
        coupling: float,
        density: float = 1.0,
        num_modes: int = 1,
        depth: int = 1,
        decay_rates=None,
        optimize_rates: bool = True,
        seed_parameters=(),
        seed_base_thetas=(),
        restarts: int = 8,
        seed=None,
        maxiter: int = 600,
        method: str = "L-BFGS-B",
        regularization: float = 1.0e-9,
        density_gauge_penalty: float = 1.0e-3,
        rate_bounds=None,
        initial_scales=(0.03, 0.06, 0.1, 0.2, 0.4),
        tie_scale: float = 0.05,
        rate_jitter: float = 0.35,
    ):
        """Optimize a true infinite-continuum fixed-density cLETTA ansatz.

        The variational state is an infinite uniform cMPS with a structured
        exponential-memory auxiliary space.  No finite real-space grid or
        finite chain length is introduced.  The returned object is the enlarged
        cLETTA cMPS; its ``cletta_*`` fields retain the base state, tie
        matrices, decay rates, hierarchy depth, and packed variational
        parameters.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scipy is required for cLETTA optimization.") from exc

        bond_dim = int(bond_dim)
        num_modes = int(num_modes)
        depth = int(depth)
        target_density = float(density)
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive.")
        if num_modes < 1:
            raise ValueError("num_modes must be positive.")
        if depth < 0:
            raise ValueError("depth must be non-negative.")
        if target_density <= 0.0:
            raise ValueError("density must be positive.")

        if decay_rates is None:
            rate_scale = max(target_density, np.sqrt(abs(float(coupling)) * target_density), 1.0e-8)
            if num_modes == 1:
                reference_rates = np.array([rate_scale], dtype=float)
            else:
                reference_rates = rate_scale * np.geomspace(0.25, 4.0, num_modes)
        else:
            reference_rates = np.atleast_1d(np.asarray(decay_rates, dtype=float))
            if reference_rates.size != num_modes:
                raise ValueError("decay_rates must contain num_modes values.")
            if np.any(~np.isfinite(reference_rates)) or np.any(reference_rates <= 0.0):
                raise ValueError("decay_rates must be finite and positive.")

        if rate_bounds is None:
            lower_rate = 1.0e-4 * target_density
            upper_rate = 1.0e3 * max(target_density, abs(float(coupling)), 1.0)
        else:
            lower_rate, upper_rate = (float(rate_bounds[0]), float(rate_bounds[1]))
            if not (0.0 < lower_rate < upper_rate):
                raise ValueError("rate_bounds must satisfy 0 < lower < upper.")

        base_size = canonical_parameter_size(bond_dim)
        tie_size = num_modes * bond_dim * bond_dim
        full_size = base_size + tie_size + (num_modes if optimize_rates else 0)
        rng = np.random.default_rng(seed)

        def pack(base_theta, ties, rates):
            pieces = [
                np.asarray(base_theta, dtype=float).ravel(),
                np.asarray(ties, dtype=float).reshape(-1),
            ]
            if optimize_rates:
                clipped = np.clip(np.asarray(rates, dtype=float), lower_rate, upper_rate)
                pieces.append(np.log(clipped))
            return np.concatenate(pieces)

        def unpack(parameters):
            parameters = np.asarray(parameters, dtype=float)
            if parameters.size != full_size:
                raise ValueError(f"parameter size {parameters.size} does not match {full_size}.")
            base_theta = parameters[:base_size]
            offset = base_size
            ties = parameters[offset : offset + tie_size].reshape(num_modes, bond_dim, bond_dim)
            offset += tie_size
            if optimize_rates:
                rates = np.exp(np.clip(parameters[offset : offset + num_modes], np.log(lower_rate), np.log(upper_rate)))
            else:
                rates = reference_rates
            return base_theta, ties, rates

        def build_state(parameters):
            base_theta, ties, rates = unpack(parameters)
            base = cls.from_canonical_parameters(base_theta, bond_dim)
            state = base.cletta_memory_state(ties, rates, depth=depth)
            state.cletta_parameters = np.asarray(parameters, dtype=float).copy()
            return state

        candidates = [np.asarray(theta, dtype=float) for theta in seed_parameters]
        base_candidates = [np.asarray(theta, dtype=float) for theta in seed_base_thetas]
        if bond_dim == 1:
            product = pack_canonical_parameters([], np.array([[np.sqrt(target_density)]]))
            base_candidates.insert(0, product)
        scales = tuple(float(value) for value in initial_scales) or (0.1,)
        while len(candidates) < int(restarts):
            if base_candidates:
                base_theta = base_candidates.pop(0)
            else:
                scale = scales[len(candidates) % len(scales)]
                base_theta = cls.random_canonical_parameters(bond_dim, rng=rng, scale=scale)
            ties = float(tie_scale) * rng.normal(size=(num_modes, bond_dim, bond_dim))
            rates = reference_rates * np.exp(float(rate_jitter) * rng.normal(size=num_modes))
            candidates.append(pack(base_theta, ties, rates))

        evaluations = 0

        def objective(parameters):
            nonlocal evaluations
            evaluations += 1
            parameters = np.asarray(parameters, dtype=float)
            if not np.all(np.isfinite(parameters)):
                return 1.0e30
            try:
                state = build_state(parameters)
                values = state.lieb_liniger_fixed_density_observables(
                    coupling=coupling,
                    density=target_density,
                    canonical=False,
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError, OverflowError):
                return 1.0e30
            energy = float(values["energy_density"])
            raw_density = float(values["raw_density"])
            if not np.isfinite(energy) or not np.isfinite(raw_density) or raw_density <= 0.0:
                return 1.0e30
            gauge = float(density_gauge_penalty) * float(np.log(raw_density / target_density) ** 2)
            regularizer = float(regularization) * float(np.dot(parameters, parameters))
            return energy + gauge + regularizer

        bounds = None
        if optimize_rates:
            bounds = [(None, None)] * (base_size + tie_size)
            bounds.extend([(np.log(lower_rate), np.log(upper_rate))] * num_modes)

        best = None
        for theta0 in candidates:
            result = minimize(
                objective,
                np.asarray(theta0, dtype=float),
                method=method,
                bounds=bounds,
                options={"maxiter": int(maxiter), "maxls": 80},
            )
            if best is None or float(result.fun) < float(best.fun):
                best = result

        if best is None:  # pragma: no cover
            raise RuntimeError("cLETTA optimization did not produce a candidate.")

        state = build_state(best.x)
        values = state.lieb_liniger_fixed_density_observables(
            coupling=coupling,
            density=target_density,
            canonical=False,
        )
        state.energy = values["energy_density"]
        state.density = values["density"]
        state.kinetic = values["kinetic"]
        state.contact = values["contact"]
        state.raw_density = values["raw_density"]
        state.scale = values["scale"]
        state.success = bool(best.success)
        state.message = str(best.message)
        state.nfev = int(evaluations)
        state.algorithm = f"fixed-density-cletta-scipy-{method}"
        return state

    @classmethod
    def optimize_lieb_liniger(
        cls,
        *,
        bond_dim: int,
        coupling: float,
        mu: float,
        seed_thetas=(),
        restarts: int = 4,
        seed=None,
        maxiter: int = 300,
        method: str = "L-BFGS-B",
        regularization: float = 1.0e-10,
    ):
        """Optimize a real left-canonical cMPS for Lieb-Liniger grand energy."""
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scipy is required for cMPS optimization.") from exc

        rng = np.random.default_rng(seed)
        candidates = [np.asarray(theta, dtype=float) for theta in seed_thetas]
        while len(candidates) < int(restarts):
            candidates.append(
                cls.random_canonical_parameters(
                    bond_dim,
                    rng=rng,
                    scale=0.25,
                )
            )

        evaluations = 0

        def objective(theta):
            nonlocal evaluations
            evaluations += 1
            try:
                state = cls.from_canonical_parameters(theta, bond_dim)
                values = state.lieb_liniger_observables(coupling=coupling, mu=mu)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return 1.0e30
            energy = values["energy_density"]
            if not np.isfinite(energy):
                return 1.0e30
            return energy + float(regularization) * float(np.dot(theta, theta))

        best = None
        for theta0 in candidates:
            result = minimize(
                objective,
                np.asarray(theta0, dtype=float),
                method=method,
                options={"maxiter": int(maxiter)},
            )
            if best is None or float(result.fun) < float(best.fun):
                best = result

        state = cls.from_canonical_parameters(best.x, bond_dim)
        values = state.lieb_liniger_observables(coupling=coupling, mu=mu)
        state.energy = values["energy_density"]
        state.density = values["density"]
        state.kinetic = values["kinetic"]
        state.contact = values["contact"]
        state.success = bool(best.success)
        state.message = str(best.message)
        state.nfev = int(evaluations)
        state.algorithm = f"canonical-scipy-{method}"
        return state

    @classmethod
    def optimize_lieb_liniger_fixed_density(
        cls,
        *,
        bond_dim: int,
        coupling: float,
        density: float = 1.0,
        seed_thetas=(),
        restarts: int = 8,
        seed=None,
        maxiter: int = 1200,
        method: str = "L-BFGS-B",
        regularization: float = 0.0,
        density_gauge_penalty: float = 1.0e-3,
        use_jax: bool = True,
        initial_scales=(0.03, 0.06, 0.1, 0.2, 0.4),
    ):
        """Optimize the fixed-density Lieb-Liniger cMPS energy.

        The physical objective is invariant under the continuum scale
        transformation ``Q -> s Q`` and ``R -> sqrt(s) R``.  The small
        ``density_gauge_penalty`` fixes that flat representative by nudging the
        raw cMPS density toward the target density without changing the
        variational energy being reported.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scipy is required for cMPS optimization.") from exc

        bond_dim = int(bond_dim)
        target_density = float(density)
        if target_density <= 0.0:
            raise ValueError("density must be positive.")
        rng = np.random.default_rng(seed)
        candidates = [np.asarray(theta, dtype=float) for theta in seed_thetas]
        scales = tuple(float(value) for value in initial_scales) or (0.25,)
        while len(candidates) < int(restarts):
            scale = scales[len(candidates) % len(scales)]
            candidates.append(cls.random_canonical_parameters(bond_dim, rng=rng, scale=scale))

        if use_jax:
            try:
                best, best_theta, evaluations = cls._optimize_lieb_liniger_fixed_density_jax(
                    bond_dim=bond_dim,
                    coupling=float(coupling),
                    density=target_density,
                    candidates=candidates,
                    method=method,
                    maxiter=maxiter,
                    regularization=regularization,
                    density_gauge_penalty=density_gauge_penalty,
                )
            except ImportError:
                best = None
            if best is not None:
                state = cls.from_canonical_parameters(best_theta, bond_dim)
                values = state.lieb_liniger_fixed_density_observables(
                    coupling=coupling,
                    density=target_density,
                )
                state.energy = values["energy_density"]
                state.density = values["density"]
                state.kinetic = values["kinetic"]
                state.contact = values["contact"]
                state.raw_density = values["raw_density"]
                state.scale = values["scale"]
                state.success = bool(best.success)
                state.message = str(best.message)
                state.nfev = int(evaluations)
                state.algorithm = f"fixed-density-jax-{method}"
                return state

        evaluations = 0

        def objective(theta):
            nonlocal evaluations
            evaluations += 1
            theta = np.asarray(theta, dtype=float)
            try:
                state = cls.from_canonical_parameters(theta, bond_dim)
                values = state.lieb_liniger_fixed_density_observables(
                    coupling=coupling,
                    density=target_density,
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return 1.0e30
            energy = values["energy_density"]
            raw_density = values["raw_density"]
            if not np.isfinite(energy) or raw_density <= 0.0:
                return 1.0e30
            gauge = float(density_gauge_penalty) * float(np.log(raw_density / target_density) ** 2)
            return energy + gauge + float(regularization) * float(np.dot(theta, theta))

        best = None
        for theta0 in candidates:
            result = minimize(
                objective,
                np.asarray(theta0, dtype=float),
                method=method,
                options={"maxiter": int(maxiter)},
            )
            if best is None or float(result.fun) < float(best.fun):
                best = result

        state = cls.from_canonical_parameters(best.x, bond_dim)
        values = state.lieb_liniger_fixed_density_observables(
            coupling=coupling,
            density=target_density,
        )
        state.energy = values["energy_density"]
        state.density = values["density"]
        state.kinetic = values["kinetic"]
        state.contact = values["contact"]
        state.raw_density = values["raw_density"]
        state.scale = values["scale"]
        state.success = bool(best.success)
        state.message = str(best.message)
        state.nfev = int(evaluations)
        state.algorithm = f"fixed-density-scipy-{method}"
        return state

    @classmethod
    def _optimize_lieb_liniger_fixed_density_jax(
        cls,
        *,
        bond_dim: int,
        coupling: float,
        density: float,
        candidates,
        method: str,
        maxiter: int,
        regularization: float,
        density_gauge_penalty: float,
    ):
        from scipy.optimize import minimize

        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover
            raise ImportError("jax is not available.") from exc

        jax.config.update("jax_enable_x64", True)
        dim = int(bond_dim)
        pairs = skew_pairs(dim)
        basis = np.zeros((len(pairs), dim, dim), dtype=float)
        for index, (row, col) in enumerate(pairs):
            basis[index, row, col] = 1.0
            basis[index, col, row] = -1.0
        basis = jnp.asarray(basis, dtype=jnp.float64)
        eye = jnp.eye(dim, dtype=jnp.float64)
        trace_row = eye.reshape(-1)
        rhs = jnp.zeros((dim * dim,), dtype=jnp.float64).at[0].set(1.0)
        coupling_value = float(coupling)
        target_density = float(density)
        regularization_value = float(regularization)
        gauge_penalty_value = float(density_gauge_penalty)

        def unpack(theta):
            a = jnp.tensordot(theta[: len(pairs)], basis, axes=(0, 0))
            r = theta[len(pairs) :].reshape((dim, dim))
            q = a - 0.5 * (r.T @ r)
            return q, r

        @jax.jit
        def observables(theta):
            q, r = unpack(theta)
            transfer = jnp.kron(q, eye) + jnp.kron(eye, q) + jnp.kron(r, r)
            matrix = transfer.at[0, :].set(trace_row)
            rho = jnp.linalg.solve(matrix, rhs).reshape((dim, dim))
            rho = 0.5 * (rho + rho.T)
            commutator = q @ r - r @ q
            rr = r @ r
            raw_density = jnp.trace(r @ rho @ r.T)
            kinetic = jnp.trace(commutator @ rho @ commutator.T)
            contact = jnp.trace(rr @ rho @ rr.T)
            return raw_density, kinetic, contact

        @jax.jit
        def objective(theta):
            raw_density, kinetic, contact = observables(theta)
            safe_density = jnp.maximum(raw_density, 1.0e-12)
            scale = target_density / safe_density
            energy = scale**3 * kinetic + coupling_value * scale**2 * contact
            gauge = gauge_penalty_value * jnp.log(safe_density / target_density) ** 2
            reg = regularization_value * jnp.dot(theta, theta)
            bad = jnp.logical_or(raw_density <= 1.0e-12, jnp.logical_not(jnp.isfinite(energy)))
            return jnp.where(bad, 1.0e30, energy + gauge + reg)

        value_and_grad = jax.jit(jax.value_and_grad(objective))
        evaluations = 0

        def value_gradient(theta):
            nonlocal evaluations
            evaluations += 1
            theta_array = np.asarray(theta, dtype=float)
            value, gradient = value_and_grad(jnp.asarray(theta_array, dtype=jnp.float64))
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                return 1.0e30, np.zeros_like(theta_array)
            return value, gradient

        best = None
        for theta0 in candidates:
            cache = {"theta": None, "value": None, "gradient": None}

            def cached(theta):
                theta_array = np.asarray(theta, dtype=float)
                if cache["theta"] is not None and np.array_equal(theta_array, cache["theta"]):
                    return cache["value"], cache["gradient"]
                value, gradient = value_gradient(theta_array)
                cache["theta"] = theta_array.copy()
                cache["value"] = value
                cache["gradient"] = gradient
                return value, gradient

            result = minimize(
                lambda theta: cached(theta)[0],
                np.asarray(theta0, dtype=float),
                jac=lambda theta: cached(theta)[1],
                method=method,
                options={"maxiter": int(maxiter), "maxls": 80},
            )
            if best is None or float(result.fun) < float(best.fun):
                best = result

        return best, np.asarray(best.x, dtype=float), evaluations


CMPS = ContinuousMPS
