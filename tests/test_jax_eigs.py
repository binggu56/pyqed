import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pyqed.jax_eigs import dominant_eig, dominant_eigval


def _matrix(parameter):
    return jnp.asarray([[1.0 + parameter, 0.2], [0.0, 0.5]], dtype=jnp.complex128)


def test_jax_dominant_eigenpair_matches_dominant_dense_pair():
    matrix = _matrix(jnp.asarray(0.0))
    right0 = jnp.ones(2)
    left0 = jnp.ones(2)
    value, left, right = dominant_eig(
        lambda vector: matrix @ vector,
        lambda vector: matrix.T @ vector,
        right0,
        left0,
        iterations=128,
        shift=2.0,
    )

    np.testing.assert_allclose(np.real(np.asarray(value)), 1.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(jnp.vdot(left, right)), 1.0, atol=1.0e-7)
    np.testing.assert_allclose(
        np.asarray(matrix @ right - value * right),
        np.zeros(2),
        atol=1.0e-6,
    )


def test_jax_dominant_eigenvalue_is_autodiffable():
    right0 = jnp.ones(2)
    left0 = jnp.ones(2)

    def objective(parameter):
        matrix = _matrix(parameter)
        value = dominant_eigval(
            lambda vector: matrix @ vector,
            lambda vector: matrix.T @ vector,
            right0,
            left0,
            iterations=128,
            shift=2.0,
        )
        return jnp.real(value)

    gradient = jax.grad(objective)(jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(gradient), 1.0, atol=1.0e-5)
