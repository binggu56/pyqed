"""Wilson-chain mappings for impurity models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WilsonChain:
    """Finite bosonic Wilson-chain representation of a star bath."""

    onsite: np.ndarray
    hopping: np.ndarray
    impurity_coupling: float
    star_frequencies: np.ndarray | None = None
    star_couplings: np.ndarray | None = None
    star_to_chain: np.ndarray | None = None

    def __post_init__(self):
        self.onsite = np.asarray(self.onsite, dtype=float)
        self.hopping = np.asarray(self.hopping, dtype=float)
        self.impurity_coupling = float(self.impurity_coupling)
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
    def from_star(cls, frequencies, couplings, *, method: str = "lanczos"):
        """Build a Wilson chain from star-bath frequencies and couplings."""
        onsite, hopping, impurity_coupling, transform = star_to_wilson_chain(
            frequencies,
            couplings,
            method=method,
        )
        return cls(
            onsite=onsite,
            hopping=hopping,
            impurity_coupling=impurity_coupling,
            star_frequencies=frequencies,
            star_couplings=couplings,
            star_to_chain=transform,
        )


def star_to_wilson_chain(frequencies, couplings, *, method: str = "lanczos"):
    """Transform a star bath to Wilson-chain parameters.

    Parameters
    ----------
    frequencies
        Star-bath oscillator frequencies.
    couplings
        Star-bath impurity couplings.
    method
        ``"lanczos"`` uses explicit reorthogonalized Lanczos tridiagonalization.
        ``"householder"`` uses a dense Householder tridiagonalization of the
        impurity-plus-star matrix.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    couplings = np.asarray(couplings, dtype=float)
    _validate_star_bath(frequencies, couplings)

    key = str(method).lower()
    if key in {"lanczos", "recursive"}:
        return _star_to_wilson_chain_lanczos(frequencies, couplings)
    if key in {"householder", "dense"}:
        return _star_to_wilson_chain_householder(frequencies, couplings)
    raise ValueError("method must be 'lanczos' or 'householder'.")


def quadrature_star_bath(spectral_density, support, nmodes: int):
    """Discretize a continuum spectral density by Gauss-Legendre quadrature.

    The convention used here matches the spin-boson helpers in this package:
    a discrete star bath represents ``J(omega)`` by
    ``sum_k g_k**2 delta(omega - omega_k)``.
    """
    nmodes = int(nmodes)
    if nmodes < 1:
        raise ValueError("nmodes must be positive.")
    if len(support) != 2:
        raise ValueError("support must be a pair ``(omega_min, omega_max)``.")
    omega_min, omega_max = map(float, support)
    if not omega_max > omega_min:
        raise ValueError("support upper bound must be larger than lower bound.")

    points, weights = np.polynomial.legendre.leggauss(nmodes)
    frequencies = 0.5 * (omega_max - omega_min) * points + 0.5 * (omega_max + omega_min)
    weights = 0.5 * (omega_max - omega_min) * weights
    try:
        density = np.asarray(spectral_density(frequencies), dtype=float)
    except Exception:
        density = np.asarray([spectral_density(omega) for omega in frequencies], dtype=float)
    if density.shape != frequencies.shape:
        raise ValueError("spectral_density must return one value per frequency.")
    if np.any(density < -1e-14):
        raise ValueError("spectral_density must be non-negative on the support.")
    couplings = np.sqrt(weights * np.maximum(density, 0.0))
    active = couplings > 1e-14 * max(1.0, float(np.max(couplings, initial=0.0)))
    if not np.any(active):
        raise ValueError("spectral_density has zero weight on the requested support.")
    return frequencies[active], couplings[active]


def orthogonal_polynomial_chain(
    spectral_density,
    *,
    support,
    nmodes: int,
    method: str = "lanczos",
    quadrature_order: int | None = None,
) -> WilsonChain:
    """Build a finite orthogonal-polynomial chain for real-time dynamics.

    This first implementation obtains the recurrence coefficients by applying
    the existing star-to-chain tridiagonalization to a Gauss quadrature of the
    continuum measure ``J(omega) d omega``.
    """
    nmodes = int(nmodes)
    if nmodes < 1:
        raise ValueError("nmodes must be positive.")
    quadrature_order = _default_quadrature_order(nmodes, quadrature_order)
    frequencies, couplings = quadrature_star_bath(
        spectral_density,
        support,
        quadrature_order,
    )
    onsite, hopping, impurity_coupling, transform = star_to_wilson_chain(
        frequencies,
        couplings,
        method=method,
    )
    return WilsonChain(
        onsite=onsite[:nmodes],
        hopping=hopping[: max(0, nmodes - 1)],
        impurity_coupling=impurity_coupling,
        star_frequencies=frequencies,
        star_couplings=couplings,
        star_to_chain=transform[:nmodes],
    )


def _default_quadrature_order(nmodes: int, quadrature_order: int | None = None) -> int:
    if quadrature_order is None:
        return max(2 * int(nmodes) + 1, int(nmodes) + 1)
    quadrature_order = int(quadrature_order)
    if quadrature_order <= int(nmodes):
        raise ValueError("quadrature_order must be larger than nmodes for OP-chain recurrence.")
    return quadrature_order


def _validate_star_bath(frequencies: np.ndarray, couplings: np.ndarray):
    if frequencies.ndim != 1 or couplings.ndim != 1:
        raise ValueError("frequencies and couplings must be one-dimensional.")
    if len(frequencies) != len(couplings):
        raise ValueError("frequencies and couplings must have the same length.")


def _star_to_wilson_chain_lanczos(frequencies: np.ndarray, couplings: np.ndarray):
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


def _star_to_wilson_chain_householder(frequencies: np.ndarray, couplings: np.ndarray):
    if len(frequencies) == 0:
        return np.array([]), np.array([]), 0.0, np.zeros((0, 0))
    if np.linalg.norm(couplings) <= 0.0:
        raise ValueError("at least one star coupling must be nonzero.")

    matrix = np.zeros((len(frequencies) + 1, len(frequencies) + 1), dtype=float)
    matrix[1:, 1:] = np.diag(frequencies)
    matrix[0, 1:] = couplings
    matrix[1:, 0] = couplings

    diagonal, offdiagonal, transform = _householder_tridiagonal(matrix)
    star_transform = transform[1:, 1:].T
    return diagonal[1:], offdiagonal[1:], float(offdiagonal[0]), star_transform


def _householder_tridiagonal(matrix: np.ndarray):
    """Return diagonal/offdiagonal and transform for a real symmetric matrix."""
    a = np.asarray(matrix, dtype=float).copy()
    n = len(a)
    for k in range(n - 2):
        u = a[k + 1 : n, k]
        u_norm = float(np.linalg.norm(u))
        if u_norm == 0.0:
            continue
        if u[0] < 0.0:
            u_norm = -u_norm
        u[0] += u_norm
        h = float(np.dot(u, u) / 2.0)
        if h == 0.0:
            continue
        v = np.dot(a[k + 1 : n, k + 1 : n], u) / h
        g = float(np.dot(u, v) / (2.0 * h))
        v -= g * u
        a[k + 1 : n, k + 1 : n] -= np.outer(v, u) + np.outer(u, v)
        a[k, k + 1] = -u_norm

    transform = np.eye(n)
    for k in range(n - 2):
        u = a[k + 1 : n, k]
        h = float(np.dot(u, u) / 2.0)
        if h == 0.0:
            continue
        v = np.dot(transform[1:n, k + 1 : n], u) / h
        transform[1:n, k + 1 : n] -= np.outer(v, u)

    return np.diagonal(a), np.diagonal(a, 1), transform


__all__ = [
    "WilsonChain",
    "orthogonal_polynomial_chain",
    "quadrature_star_bath",
    "star_to_wilson_chain",
]
