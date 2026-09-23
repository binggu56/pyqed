"""Full molecular BSE integration: actions, physical metrics and failure gates."""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest

bse = importlib.import_module('pyqed.gw.bse')


def model(factorized=True):
    rng = np.random.default_rng(731)
    factors = rng.normal(size=(7, 6, 6))*.02
    factors = (factors+factors.transpose(0, 2, 1))/2
    m = rng.normal(size=(6, 6, 5))*.01
    m = (m+m.transpose(1, 0, 2))/2
    return SimpleNamespace(nocc=2, nso=6, e_qp=None, e_mf=np.arange(6.),
        _M=m, e_rpa=np.arange(1., 6.), _pair_factors=factors if factorized else None,
        eri=None if factorized else np.einsum('Pij,Pkl->ijkl', factors, factors))


@pytest.mark.parametrize('factorized', [False, True])
def test_full_batched_complex_action_and_public_driver(factorized):
    gw = model(factorized)
    rng = np.random.default_rng(91)
    block = rng.normal(size=(16, 11))+1j*rng.normal(size=(16, 11))
    expected = np.column_stack([bse._bse_full_matvec(gw, v) for v in block.T])
    np.testing.assert_allclose(bse._bse_full_matmat(gw, block, 3), expected, atol=1e-13)
    assert bse._bse_full_matmat(gw, block[:, :0], 3).shape == (16, 0)
    h = np.column_stack([bse._bse_full_matvec(gw, v) for v in np.eye(16)])
    expected = np.sort(np.linalg.eigvals(h).real)[8:11]
    for batch in [None, 3]:
        solver = bse.BSE.__new__(bse.BSE)
        solver.__dict__.update(gw.__dict__)
        assert solver.run(nroots=3, use_qp=False, low_rank=True, eigensolver='davidson',
                          max_space=16, batch_columns=batch, tol=1e-10) is solver
        np.testing.assert_allclose(solver.e, expected, atol=1e-10)
        assert solver.info['eigensolver'] == 'davidson'
        assert solver.info['converged']
        assert max(np.linalg.norm(h@solver.xy-solver.xy*solver.e, axis=0)) <= 1e-10
        np.testing.assert_allclose(solver.x.T@solver.x-solver.y.T@solver.y, np.eye(3), atol=1e-10)
        if batch:
            assert solver.info['davidson']['matmat_callback_calls'] > 0
            assert solver.info['davidson']['matvec_callback_calls'] == 0
    with pytest.raises(ValueError, match='low_rank'):
        solver.run(low_rank=False, eigensolver='davidson')


def operator_model(monkeypatch, a, coupling):
    h = np.block([[a, coupling], [-coupling, -a]])
    gw = SimpleNamespace(nocc=1, nso=len(a)+1, e_rpa=np.ones(1), _M=np.zeros(1))
    monkeypatch.setattr(bse, '_bse_full_matvec', lambda gw, v: h@v)
    monkeypatch.setattr(bse, '_bse_full_matmat', lambda gw, v, batch: h@v)
    monkeypatch.setattr(bse, '_bse_tda_diag', lambda gw: np.diag(a))
    return gw, h


@pytest.mark.parametrize('solver', ['arpack', 'davidson'])
def test_small_gap_and_cluster_against_dense(monkeypatch, solver):
    q, _ = np.linalg.qr(np.random.default_rng(58).normal(size=(6, 6)))
    a = (q*np.array([.1000000005, .3, .3+1e-9, 1., 2., 3.]))@q.T
    coupling = (q*np.array([.1, .02, .02, .05, .05, .05]))@q.T
    gw, h = operator_model(monkeypatch, a, coupling)
    w, v, info = bse.solve_bse(gw, nroots=3, tol=1e-9, eigensolver=solver,
                              max_cycle=300, max_space=12, return_info=True)
    reference = np.sort(np.linalg.eigvals(h).real)[6:9]
    np.testing.assert_allclose(w, reference, atol=1e-9, rtol=0)
    assert max(np.linalg.norm(h@v-v*w, axis=0)) < 1e-9
    assert info['metric_error'] < 1e-8


@pytest.mark.parametrize('kind', ['imaginary', 'negative_metric'])
def test_unstable_selected_modes_rejected(monkeypatch, kind):
    a = np.eye(4) if kind == 'imaginary' else np.diag([-1., 2., 3., 4.])
    coupling = 2*np.eye(4) if kind == 'imaginary' else np.zeros((4, 4))
    gw, _ = operator_model(monkeypatch, a, coupling)
    with pytest.raises(RuntimeError):
        bse.solve_bse(gw, nroots=1, eigensolver='davidson', max_space=8, max_cycle=20)
    assert not gw.info['converged']
    assert not hasattr(gw, 'excitation_energies')


def test_post_normalization_residual_and_nonconvergence(monkeypatch):
    gw = model()
    original = bse._metric_orthonormalize_full_bse_vectors
    def corrupt(vectors, dim):
        out = original(vectors, dim)
        out[-1, 0] += .01
        return out
    monkeypatch.setattr(bse, '_metric_orthonormalize_full_bse_vectors', corrupt)
    with pytest.raises(RuntimeError, match='checks after normalization'):
        bse.solve_bse(gw, eigensolver='davidson', nroots=2, max_space=16)
    assert not gw.info['converged']
    monkeypatch.setattr(bse, '_metric_orthonormalize_full_bse_vectors', original)
    with pytest.raises(RuntimeError, match='did not converge'):
        bse.solve_bse(gw, eigensolver='davidson', nroots=2, max_cycle=1)
    assert not gw.info['converged']


@pytest.mark.parametrize('kwargs', [dict(eigensolver='unknown'), dict(eigensolver="arpack", batch_columns=3),
    dict(eigensolver='davidson', batch_columns=1.5), dict(max_space=1.5), dict(tol=0)])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        bse.solve_bse(model(), **kwargs)


def test_one_transition_and_phase_alignment(monkeypatch):
    from pyqed import linalg
    original = linalg.davidson_nonsymmetric
    def phased(*args, **kwargs):
        w, v, info = original(*args, **kwargs)
        return w, v*np.exp(.8j), info
    monkeypatch.setattr(linalg, 'davidson_nonsymmetric', phased)
    gw, _ = operator_model(monkeypatch, np.array([[1.]]), np.array([[.2]]))
    w, info = bse.solve_bse(gw, nroots=1, eigensolver='davidson',
                            return_vectors=False, return_info=True, tol=1e-10)
    np.testing.assert_allclose(w, [np.sqrt(.96)], atol=1e-10)
    assert info['converged']
    assert np.isrealobj(gw.xy)


def test_arpack_partial_results_rejected(monkeypatch):
    from scipy.sparse import linalg
    def partial(*args, **kwargs):
        raise linalg.ArpackNoConvergence('forced incomplete solve', np.array([1.]), np.eye(16, 1))
    monkeypatch.setattr(linalg, 'eigs', partial)
    gw = model()
    with pytest.raises(linalg.ArpackNoConvergence):
        bse.solve_bse(gw, nroots=1, eigensolver="arpack")
    assert not gw.info['converged']
    assert gw.info['eigensolver'] == 'arpack'
    assert not hasattr(gw, 'excitation_energies')


@pytest.mark.parametrize('roots', [1, 2])
def test_real_degenerate_subspace_from_complex_mixtures(monkeypatch, roots):
    from pyqed import linalg
    gw, h = operator_model(monkeypatch, np.diag([1., 1., 2., 3.]), np.zeros((4, 4)))
    vectors = np.zeros((8, 2), dtype=complex)
    vectors[0] = [1., 1.]
    vectors[1] = [1j, -1j]
    vectors /= np.sqrt(2)
    monkeypatch.setattr(linalg, 'davidson_nonsymmetric',
        lambda *args, **kwargs: (np.ones(roots), vectors[:, :roots], {'converged': True}))
    w, v, info = bse.solve_bse(gw, nroots=roots, eigensolver='davidson', return_info=True)
    assert info['converged']
    assert np.isrealobj(v)
    np.testing.assert_allclose(h@v, v*w, atol=1e-12)
    np.testing.assert_allclose(v[:4].T@v[:4]-v[4:].T@v[4:], np.eye(roots), atol=1e-12)


@pytest.mark.parametrize('entry', ['function', 'method', 'run'])
def test_default_full_bse_uses_davidson(entry):
    gw = model()
    solver = bse.BSE.__new__(bse.BSE)
    solver.__dict__.update(gw.__dict__)
    solver.e_qp = gw.e_mf.copy()
    if entry == 'function':
        bse.solve_bse(solver, nroots=2)
    elif entry == 'method':
        solver.solve_bse(nroots=2)
    else:
        solver.run(nroots=2, use_qp=False)
    assert solver.info['eigensolver'] == 'davidson'
    assert solver.info['converged']
    h = np.column_stack([bse._bse_full_matvec(solver, v) for v in np.eye(16)])
    expected = np.sort(np.linalg.eigvals(h).real)[8:10]
    np.testing.assert_allclose(solver.e, expected, atol=1e-8, rtol=0)
