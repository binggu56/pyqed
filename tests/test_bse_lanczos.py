import numpy as np
import pytest
from test_bse_nonsymmetric import bse, operator_model


@pytest.mark.parametrize('batch', [None, 3, 'auto'])
def test_lanczos_dense(monkeypatch, batch):
    rng = np.random.default_rng(23)
    r = rng.normal(size=(28, 28))
    a = r@r.T/28 + np.diag(np.linspace(2, 5, 28))
    r = rng.normal(size=(28, 28))
    b = .02*(r+r.T)
    gw, h = operator_model(monkeypatch, a, b)
    e, xy, info = bse.solve_bse(gw, nroots=3, eigensolver='lanczos',
        max_space=28, max_cycle=30, tol=1e-9, batch_columns=batch, return_info=True)
    ref = np.sort(np.linalg.eigvals(h).real)
    np.testing.assert_allclose(e, ref[ref>0][:3], atol=1e-10)
    assert np.max(np.linalg.norm(h@xy-xy*e, axis=0)) < 1e-9
    np.testing.assert_allclose(xy[:28].T@xy[:28]-xy[28:].T@xy[28:], np.eye(3), atol=1e-10)
    assert info['lanczos']['history']


def test_degenerate(monkeypatch):
    a = np.diag([2., 2., 2., 3., 4., 5.])
    gw, _ = operator_model(monkeypatch, a, np.zeros_like(a))
    e, _ = bse.solve_bse(gw, nroots=3, eigensolver='lanczos', max_space=6, tol=1e-9)
    np.testing.assert_allclose(e, 2., atol=1e-10)


def test_caps(monkeypatch):
    a = np.diag(np.arange(1., 21.))
    gw, _ = operator_model(monkeypatch, a, np.zeros_like(a))
    with pytest.raises(RuntimeError, match='max_space'):
        bse.solve_bse(gw, nroots=2, eigensolver='lanczos', max_space=4)
    assert not gw.info['converged']
    assert len(gw.info['lanczos']['history']) == 2
    gw.max_memory = .0001
    with pytest.raises(MemoryError):
        bse.solve_bse(gw, nroots=2, eigensolver='lanczos')


@pytest.mark.parametrize('coupling', [3., -3.])
def test_unstable(monkeypatch, coupling):
    gw, _ = operator_model(monkeypatch, np.eye(4), coupling*np.eye(4))
    with pytest.raises(RuntimeError, match='positive definite'):
        bse.solve_bse(gw, nroots=2, eigensolver='lanczos')


def test_wide_spectrum_metric_recurrence(monkeypatch):
    rng = np.random.default_rng(91)
    q, _ = np.linalg.qr(rng.normal(size=(80, 80)))
    a = q@np.diag(np.geomspace(.1, 30., 80))@q.T
    coupling = .01*np.eye(80)
    gw, h = operator_model(monkeypatch, a, coupling)
    roots, xy = bse.solve_bse(gw, nroots=3, eigensolver='lanczos',
                             max_space=80, max_cycle=40, tol=1e-8)
    expected = np.sqrt(np.geomspace(.1, 30., 80)[:3]**2-.01**2)
    np.testing.assert_allclose(roots, expected, atol=1e-9, rtol=0)
    assert max(np.linalg.norm(h@xy-xy*roots, axis=0)) < 1e-8


def test_real_and_symmetric_restrictions(monkeypatch):
    a = np.eye(4, dtype=complex)*(1+.01j)
    gw, _ = operator_model(monkeypatch, a, np.zeros_like(a))
    with pytest.raises(ValueError, match='real operator'):
        bse.solve_bse(gw, nroots=2, eigensolver='lanczos')
    a = np.eye(4);a[0, 1] = .2
    gw, _ = operator_model(monkeypatch, a, np.zeros_like(a))
    with pytest.raises(RuntimeError, match='symmetric'):
        bse.solve_bse(gw, nroots=4, eigensolver='lanczos')
