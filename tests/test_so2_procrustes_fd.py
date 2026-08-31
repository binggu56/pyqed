import numpy as np

from examples.ldr.so2_procrustes_fd import local_derivative


def test_local_derivative_is_quadratic_exact_on_nonuniform_grid():
    points = np.asarray([-0.8, -0.2, 0.1, 0.7, 1.4])
    weights = np.asarray([0.2, 0.4, 0.5, 0.35, 0.1])
    values = np.sqrt(weights) * (0.3 + 0.7 * points - 0.4 * points**2)
    expected = np.sqrt(weights) * (0.7 - 0.8 * points)
    np.testing.assert_allclose(local_derivative(points, weights) @ values, expected)


def test_five_point_derivative_is_quartic_exact():
    points = np.asarray([-1.0, -0.6, -0.1, 0.5, 1.3, 1.8])
    weights = np.linspace(0.2, 0.7, len(points))
    polynomial = 0.2 - points + 0.3 * points**2 + 0.1 * points**4
    expected = -1.0 + 0.6 * points + 0.4 * points**3
    encoded = np.sqrt(weights) * polynomial
    np.testing.assert_allclose(
        local_derivative(points, weights, 5) @ encoded,
        np.sqrt(weights) * expected,
        atol=1.0e-12,
    )
