import numpy as np
import pytest
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


@pytest.mark.parametrize(
    "method",
    ["krylov", "exp_krylov", "expm-krylov", "exponential_krylov"],
)
def test_solve_ivp_krylov_aliases_match_direct_krylov(method):
    matrix = np.array([[-0.2, 0.1], [-0.3, -0.7]], dtype=np.complex128)
    y0 = np.array([1.0, 2.0], dtype=np.complex128)
    t_eval = np.linspace(0.25, 0.75, 6)
    rhs_times = []

    def rhs(t, y):
        rhs_times.append(t)
        return matrix @ y

    actual = solve_ivp(
        rhs,
        y0,
        t_eval,
        method=method,
        krylov_dim=2,
    )
    expected = krylov(lambda y: matrix @ y, y0, t_eval, krylov_dim=2)

    assert actual.success
    np.testing.assert_allclose(actual.y, expected.y, atol=1.0e-14)
    assert rhs_times
    assert set(rhs_times) == {t_eval[0]}


@pytest.mark.parametrize(
    "t_eval, message",
    [
        ([0.0, 0.0, 1.0], "strictly increasing"),
        ([0.0, 1.0, 0.5], "strictly increasing"),
        ([0.0, np.nan, 1.0], "finite"),
        ([0.0, np.inf, 1.0], "finite"),
    ],
    ids=["duplicate", "decreasing", "nan", "infinity"],
)
def test_krylov_rejects_invalid_t_eval(t_eval, message):
    with pytest.raises(ValueError, match=message):
        krylov(lambda y: y, [1.0], t_eval)


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
