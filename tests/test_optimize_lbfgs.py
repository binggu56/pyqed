"""Analytic Stiefel objectives with orbital-like curvature scales."""

import numpy as np
import pytest

from pyqed.optimize import grad, minimize, norm


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_lbfgs_converges_scaled_rayleigh_quotient(scale):
    matrix = scale * np.diag([0.0, 1.0, 10.0, 100.0])
    initial = np.array([[1.0], [0.1], [0.02], [0.001]])
    initial /= np.linalg.norm(initial)

    def energy(u):
        return float((u.T @ matrix @ u).item())

    def gradient(u):
        return 2 * matrix @ u

    orbitals, value = minimize(
        energy, initial, gradient_fn=gradient, algorithm="LBFGS",
        max_iterations=100, epsilon=1e-7, tau=1.0,
    )
    np.testing.assert_allclose(orbitals.T @ orbitals, np.eye(1), atol=1e-12)
    assert value < 1e-12
    assert norm(grad(orbitals, gradient(orbitals))) < 1e-7


@pytest.mark.parametrize("amplitude", [1e-6, 1e-7])
def test_lbfgs_refines_small_steps(amplitude):
    matrix = np.diag([0.0, 1.0, 10.0, 1000.0])
    initial = np.array([[1.0], [amplitude], [amplitude], [amplitude]])
    initial /= np.linalg.norm(initial)
    gradient = lambda u: 2 * matrix @ u
    orbitals, value = minimize(
        lambda u: float((u.T @ matrix @ u).item()), initial,
        gradient_fn=gradient, algorithm="LBFGS", max_iterations=100,
        epsilon=1e-9, tau=1.0,
    )
    assert value < 1e-20
    assert norm(grad(orbitals, gradient(orbitals))) < 1e-9
