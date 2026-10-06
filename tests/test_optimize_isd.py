"""Reference implicit updates and analytic constrained minima for ISD."""
import numpy as np
import pytest
from pyqed.optimize import isd_step, minimize, project


@pytest.mark.parametrize('complex_orbitals', [False, True])
def test_isd_step_matches_implicit_system(complex_orbitals):
    rng = np.random.default_rng(21)
    x, g = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    if complex_orbitals:
        x = x + 1j * rng.normal(size=x.shape)
        g = g + 1j * rng.normal(size=g.shape)
    x = project(x)
    a = g @ x.conj().T - x @ g.conj().T
    y = isd_step(x, g, .07)
    np.testing.assert_allclose(y, project(np.linalg.solve(np.eye(7) + .07*a, x)), atol=1e-12)
    np.testing.assert_allclose(y.conj().T @ y, np.eye(3), atol=1e-12)
    tangent = g - x @ (g.conj().T @ x)
    np.testing.assert_allclose((isd_step(x, g, 1e-7)-x)/1e-7, -tangent, atol=3e-6)


def test_isd_converges_anisotropic_rayleigh_quotient():
    matrix = np.diag([0., 1., 10., 100.])
    initial = project(np.array([[1.], [.1], [.02], [.001]]))
    g = lambda u: 2 * matrix @ u
    u, energy = minimize(lambda u: float((u.T @ matrix @ u).item()), initial,
                        gradient_fn=g, algorithm='ISD', tau=1.,
                        max_iterations=500, epsilon=1e-7)
    assert energy < 1e-14
    assert np.linalg.norm(g(u) - u @ (g(u).T @ u)) < 1e-7


def test_isd_rejects_restricted_projection():
    with pytest.raises(ValueError, match='custom tangent projection'):
        minimize(lambda u: 0., np.eye(2), algorithm='ISD',
                 gradient_fn=lambda u: u, projection_fn=lambda u, g: g)


def test_isd_armijo_accepts_rotation_inside_frame():
    # The exact polar update is a two-dimensional rotation for this objective.
    g = np.array([[0., 1.], [-1., 0.]])
    step = .3
    initial = np.eye(2)
    expected = (initial - 2*step*g) / np.sqrt(1 + 4*step**2)
    orbitals, energy = minimize(
        lambda x: float(np.vdot(g, x).real), initial,
        gradient_fn=lambda x: g, algorithm='ISD', tau=step, max_iterations=1,
    )
    np.testing.assert_allclose(orbitals, expected, atol=1e-13)
    np.testing.assert_allclose(energy, -4*step/np.sqrt(1+4*step**2), atol=1e-13)
