"""Structured response Davidson against full dense XY references."""
import numpy as np
import pytest
from test_bse_nonsymmetric import bse, operator_model


@pytest.mark.parametrize('batch', [None, 3, 'auto'])
def test_response_dense_reference_and_restarts(monkeypatch, batch):
    rng = np.random.default_rng(498)
    n = 28
    a = np.diag(np.linspace(.3, 3., n))
    u = rng.normal(scale=.025, size=(n, n))
    a += u+u.T
    u = rng.normal(scale=.01, size=(n, n))
    coupling = u+u.T
    gw, h = operator_model(monkeypatch, a, coupling)
    roots, xy, info = bse.solve_bse(gw, nroots=3, tol=1e-9, max_cycle=200,
                                   max_space=12, batch_columns=batch, return_info=True)
    expected = np.sort(np.linalg.eigvals(h).real)[n:n+3]
    np.testing.assert_allclose(roots, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(xy[:n].T@xy[:n]-xy[n:].T@xy[n:], np.eye(3), atol=1e-10)
    assert max(np.linalg.norm(h@xy-xy*roots, axis=0)) < 1e-9
    assert info['response']['restarts'] > 0


@pytest.mark.parametrize('which', ['plus', 'minus', 'nonsymmetric'])
def test_response_rejects_invalid_projected_operator(monkeypatch, which):
    a = np.diag([1., 2., 3.])
    coupling = np.diag([-2., 0., 0.] if which=='plus' else [2., 0., 0.])
    if which=='nonsymmetric':
        coupling[:] = 0
        a[0, 1] = .3
    gw, _ = operator_model(monkeypatch, a, coupling)
    with pytest.raises((RuntimeError, ValueError), match='positive|symmetric'):
        bse.solve_bse(gw, nroots=3)
    assert not gw.info['converged']


def test_response_degeneracy_and_failures(monkeypatch):
    gw, h = operator_model(monkeypatch, np.diag([1., 1., 2., 3.]), np.zeros((4, 4)))
    roots, xy = bse.solve_bse(gw, nroots=4, tol=1e-11)
    np.testing.assert_allclose(roots, [1., 1., 2., 3.], atol=1e-12)
    np.testing.assert_allclose(h@xy, xy*roots, atol=1e-12)
    gw.max_memory = 1e-6
    with pytest.raises(MemoryError, match='workspace'):
        bse.solve_bse(gw, nroots=1)
    gw.max_memory = 4000
    with pytest.raises(ValueError, match='subspace'):
        bse.solve_bse(gw, nroots=2, max_space=2)
    a = np.diag(np.arange(1., 7.))+.03
    gw, _ = operator_model(monkeypatch, a, np.full_like(a, .01))
    with pytest.raises(RuntimeError, match='did not converge'):
        bse.solve_bse(gw, nroots=2, max_cycle=1)


def test_response_rejects_complex_operator(monkeypatch):
    gw, _ = operator_model(monkeypatch, np.eye(3, dtype=complex)*(1+.01j),
                           np.zeros((3, 3)))
    with pytest.raises(ValueError, match='real operator'):
        bse.solve_bse(gw, nroots=1)
