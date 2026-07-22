"""Differentiable matrix-free dominant eigenpairs for JAX.

This module intentionally does not import JAX at module import time.  The
solver is a fixed-iteration two-sided Krylov/power iteration: it is useful for
smooth optimization of transfer operators, while SciPy/ARPACK remains the
high-accuracy convergence reference.
"""

from __future__ import annotations


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("jax is required for dominant_eig.") from exc
    return jax, jnp


def dominant_eig(
    matvec,
    adjoint_matvec,
    right0,
    left0,
    *,
    iterations=256,
    shift=0.0,
    eps=1.0e-30,
):
    """Return a differentiable dominant left/right eigenpair.

    Parameters
    ----------
    matvec, adjoint_matvec : callable
        JAX-compatible actions of ``A`` and ``A.conj().T``.  The callbacks may
        close over variational JAX arrays and can therefore be used inside
        ``jax.jit`` and ``jax.value_and_grad``.
    right0, left0 : array_like
        Nonzero initial right and left vectors with the same dimension.
    iterations : int
        Fixed number of iterations.  Keeping this static makes the routine
        compatible with JAX transformations.
    shift : scalar
        Scalar shift used for power iteration.  Choose it so the desired
        eigenvalue has largest magnitude after shifting.  For transfer
        generators this is normally a positive real number larger than the
        spectral width.
    eps : float
        Stabilizer for vector normalization.

    Returns
    -------
    eigenvalue, left, right
        The Rayleigh eigenvalue and biorthogonally normalized vectors satisfying
        ``vdot(left, right) == 1`` up to iteration error.

    Notes
    -----
    This is an autodiff-friendly iterative eigensolver, not a certified
    Arnoldi convergence routine.  Use a SciPy ``eigs`` result to validate the
    iteration count and shift for each transfer family.
    """
    jax, jnp = _require_jax()

    right0 = jnp.asarray(right0)
    left0 = jnp.asarray(left0)
    dtype = jnp.result_type(right0, left0, jnp.complex64)
    right0 = right0.astype(dtype)
    left0 = left0.astype(dtype)
    identity_shift = jnp.asarray(shift, dtype=dtype)

    def normalize(vector):
        norm = jnp.sqrt(jnp.real(jnp.vdot(vector, vector)) + eps)
        return vector / norm

    def body(_, state):
        right, left = state
        right_next = matvec(right) + identity_shift * right
        left_next = adjoint_matvec(left) + identity_shift * left
        return normalize(right_next), normalize(left_next)

    right, left = jax.lax.fori_loop(
        0,
        int(iterations),
        body,
        (normalize(right0), normalize(left0)),
    )
    overlap = jnp.vdot(left, right)
    right = right / overlap
    eigenvalue = jnp.vdot(left, matvec(right))
    return eigenvalue, left, right


def dominant_eigval(
    matvec,
    adjoint_matvec,
    right0,
    left0,
    *,
    iterations=256,
    shift=0.0,
    eps=1.0e-30,
):
    """Convenience wrapper returning only the differentiable eigenvalue."""
    eigenvalue, _, _ = dominant_eig(
        matvec,
        adjoint_matvec,
        right0,
        left0,
        iterations=iterations,
        shift=shift,
        eps=eps,
    )
    return eigenvalue


__all__ = ["dominant_eig", "dominant_eigval"]
