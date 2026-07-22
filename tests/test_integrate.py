import numpy as np
from scipy import linalg as la

from pyqed.integrate import (
    dop853,
    expm_krylov,
    krylov,
    normalize_method,
    solve_ivp,
)


def test_normalize_method_aliases():
    assert normalize_method("dopri853") == "dop853"
    assert normalize_method("expm_krylov") == "krylov"
    assert normalize_method("RK4") == "rk4"


def test_expm_krylov_matches_dense_expm():
    matrix = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    y0 = np.array([1.0, 0.0], dtype=np.complex128)

    actual = expm_krylov(lambda y: matrix @ y, y0, 0.3, krylov_dim=2)
    expected = la.expm(0.3 * matrix) @ y0

    np.testing.assert_allclose(actual, expected, atol=1.0e-14)


def test_krylov_matches_dense_exponential_on_grid():
    matrix = np.array([[-0.2, 0.0], [0.0, -0.7]], dtype=np.complex128)
    y0 = np.array([1.0, 2.0], dtype=np.complex128)
    t_eval = np.linspace(0.0, 0.5, 6)

    sol = krylov(lambda y: matrix @ y, y0, t_eval, krylov_dim=2)

    expected = np.array([la.expm(t * matrix) @ y0 for t in t_eval]).T
    assert sol.success
    np.testing.assert_allclose(sol.y, expected, atol=1.0e-14)


def test_solve_ivp_dop853_exponential_decay():
    t_eval = np.linspace(0.0, 1.0, 6)
    y0 = np.array([1.0 + 0.0j])

    sol = solve_ivp(
        lambda _t, y: -0.5 * y,
        y0,
        t_eval,
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-13,
    )

    assert sol.success
    assert sol.n_steps > 0
    np.testing.assert_allclose(sol.y[0], np.exp(-0.5 * t_eval), atol=1.0e-11)


def test_dop853_streams_observations_without_saving_states():
    t_eval = np.linspace(0.0, 2.0, 21)
    y0 = np.array([1.0 + 0.0j])

    sol = dop853(
        lambda _t, y: -0.5 * y,
        y0,
        t_eval,
        rtol=1.0e-11,
        atol=1.0e-13,
        observer=lambda _t, y: y[0],
        save=False,
    )

    assert sol.success
    assert sol.y is None
    assert sol.n_steps > 0
    np.testing.assert_allclose(sol.observations, np.exp(-0.5 * t_eval), atol=1.0e-11)
    np.testing.assert_allclose(sol.y_final, [np.exp(-1.0)], atol=1.0e-11)
