"""Streamed BSE/TDA contractions against independent dense static kernels."""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.linalg import cho_factor
from pyqed.gw.screening import FactorizedCouplings, StaticScreening

b = importlib.import_module('pyqed.gw.bse')


def model(static=False, window=False):
    rng = np.random.default_rng(453)
    f = rng.normal(scale=.02, size=(131, 7, 7))
    f = (f+f.transpose(0, 2, 1))/2
    f = f[:, 1:6, 1:6] if window else f[:, :5, :5].copy()
    projection = rng.normal(scale=.015, size=(131, 9))
    poles = np.linspace(.4, 3., 9)
    screening = FactorizedCouplings(f, projection, poles)
    if static:
        dielectric = np.eye(len(f))+projection@projection.T
        screening = StaticScreening(f, cho_factor(dielectric))
    return SimpleNamespace(nocc=2, nso=5, e_mf=np.array([-1., -.5, .2, .5, .9]),
        e_qp=None, _pair_factors=f, _M=screening, e_rpa=poles, threads=1, max_memory=4.)


def dense_blocks(m):
    no, f = m.nocc, m._pair_factors
    w = m._M
    a = (2*np.einsum('Pai,Pbj->iajb', f[:, no:, :no], f[:, no:, :no])
         -np.einsum('Pij,Pab->iajb', f[:, :no, :no], f[:, no:, no:])
         +4*np.einsum('Pij,Pab->iajb', w.static_occupied(no), f[:, no:, no:])).reshape(6, 6)
    bb = (2*np.einsum('Pai,Pjb->iajb', f[:, no:, :no], f[:, :no, no:])
          -np.einsum('Pib,Paj->iajb', f[:, :no, no:], f[:, no:, :no])
          +4*np.einsum('Pib,Paj->iajb', w.static_ov(no), f[:, no:, :no])).reshape(6, 6)
    a += np.diag((m.e_mf[no:]-m.e_mf[:no, None]).ravel())
    return a, bb


@pytest.mark.parametrize('static', [False, True])
@pytest.mark.parametrize('window', [False, True])
@pytest.mark.parametrize('complex_trials', [False, True])
def test_actions_and_fixed_order_reduction(static, window, complex_trials, monkeypatch):
    kernel = pytest.importorskip('pyqed.gw._bse_contractions', exc_type=ModuleNotFoundError)
    if not kernel.openmp_available():
        pytest.skip('OpenMP unavailable')
    m = model(static, window)
    a, bb = dense_blocks(m)
    x = np.random.default_rng(734).normal(size=(6, 7))
    if complex_trials:
        x = x+1j*x[::-1]
    full = np.vstack((x, x[::-1]))
    monkeypatch.setattr(FactorizedCouplings, '__array__',
                        lambda *a, **kw: pytest.fail('full pole tensor materialized'))
    actions = [(b._bse_tda_matmat, x, a@x), (b._bse_b_matmat, x, bb@x),
               (b._bse_full_matmat, full, np.block([[a, bb], [-bb, -a]])@full)]
    for action, trial, expected in actions:
        reference = action(m, trial, 3)
        np.testing.assert_allclose(reference, expected, atol=1e-12, rtol=1e-12)
        for threads in [2, 4, 8]:
            m.threads = threads
            actual = action(m, trial, 3)
            np.testing.assert_array_equal(actual, reference)
            assert m.threading_info['actual_threads'] == threads
        m.threads = 1
        assert action(m, trial[:, :0], 3).shape == trial[:, :0].shape


def test_solver_roots_residuals_and_memory_bound():
    kernel = pytest.importorskip('pyqed.gw._bse_contractions', exc_type=ModuleNotFoundError)
    if not kernel.openmp_available():
        pytest.skip('OpenMP unavailable')
    m = model()
    a, bb = dense_blocks(m)
    expected = np.sort(np.linalg.eigvals(np.block([[a, bb], [-bb, -a]])).real)
    expected = expected[expected > 0][:3]
    m.threads = 8
    energies, states, info = b.solve_bse(m, nroots=3, batch_columns=3, tol=1e-10, return_info=True)
    np.testing.assert_allclose(energies, expected, atol=1e-9, rtol=0)
    assert np.max(info['residual_norms']) <= 1e-10
    m.max_memory = .0001
    m.threading_info = None
    b._bse_full_matmat(m, states, 3)
    assert m.threading_info['actual_threads'] == 1


@pytest.mark.parametrize('active', ['both', 'x', 'y'])
def test_full_action_wide_blas_products(active):
    from threadpoolctl import threadpool_limits
    kernel = pytest.importorskip('pyqed.gw._bse_contractions')
    if not kernel.openmp_available():
        pytest.skip('OpenMP unavailable')
    # Exercise BLAS-sized products, a partial auxiliary tile, and unequal
    # orbital spaces. Nonsymmetric factors check both pair orientations.
    rng = np.random.default_rng(718)
    no, nv, rank, columns = 13, 37, 35, 5
    factors = rng.normal(scale=.02, size=(rank, no+nv, no+nv))
    occupied = rng.normal(scale=.01, size=(rank, no, no))
    ov = rng.normal(scale=.01, size=(rank, no, nv))
    x, y = rng.normal(size=(2, no*nv, columns))
    if active == 'x':
        y.fill(0)
    if active == 'y':
        x.fill(0)
    with threadpool_limits(limits=1, user_api='blas'):
        def action(v, kind, threads=1):
            return kernel.action(factors, occupied, ov, v, no, kind, threads)[0]
        expected = np.vstack((action(x, 'a')+action(y, 'b'),
                              -action(x, 'b')-action(y, 'a')))
        trial = np.vstack((x, y))
        actual = action(trial, 'full')
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        for threads in [2, 4, 8]:
            np.testing.assert_array_equal(action(trial, 'full', threads), actual)


@pytest.mark.parametrize('threads', [0, -1, 1.5, True, np.bool_(True)])
def test_invalid_threads(threads):
    with pytest.raises(ValueError, match='positive integer'):
        b.BSE(None, threads=threads)


def test_missing_extension_and_unsupported_paths(monkeypatch):
    monkeypatch.setattr(b, '_bse_contractions', None)
    with pytest.raises(RuntimeError, match='with OpenMP'):
        b.BSE(None, threads=2)
    m = model()
    m.threads = 2
    with pytest.raises(NotImplementedError, match='factorized'):
        b._bse_tda_matmat(m, np.ones((6, 1)))


def test_scanner_retains_independent_thread_controls(monkeypatch):
    g = importlib.import_module('pyqed.gw.gw')
    previous = SimpleNamespace(threads=2, freq_int='exact', method='g0w0')
    mf = SimpleNamespace(e_tot=-1.)
    base = SimpleNamespace(mol=object(), _scf=mf, gw=previous, threads=4,
                           screening='TDH', eta=.001, e=None, orbital_window=(2, 1), screening_energies='qp')
    calls = {}
    class FakeGW:
        def __init__(self, mf, **kwargs):
            calls['gw'] = kwargs['threads']
            self.e_tot = mf.e_tot
        def run(self, **kwargs):
            return self
    class FakeBSE:
        def __init__(self, gw, **kwargs):
            calls['bse'] = kwargs
            self.e = np.array([.2, .3])
        def run(self, **kwargs):
            return self
    monkeypatch.setattr(g, 'GW', FakeGW)
    scanner = b.GWBSEScanner(base, nroots=2, energy='excitation', solver_cls=FakeBSE)
    monkeypatch.setattr(scanner, '_prepare_mol', lambda value: base.mol)
    monkeypatch.setattr(scanner, '_run_mf', lambda mol: mf)
    scanner(base.mol)
    assert calls == dict(gw=2, bse=dict(threads=4, screening='TDH', eta=.001,
                                      orbital_window=(2, 1), screening_energies='qp'))


@pytest.mark.parametrize('static', [False, True])
@pytest.mark.parametrize('layout', ['fortran', 'reversed'])
def test_serial_dispatch_retains_unsupported_layout_fallback(static, layout, monkeypatch):
    m = model(static)
    factors = m._pair_factors
    factors = np.asfortranarray(factors) if layout == 'fortran' else factors[:, ::-1, ::-1]
    m._pair_factors = m._M.factors = factors
    a, bb = dense_blocks(m)
    vectors = np.random.default_rng(983).normal(size=(12, 3))
    vectors = vectors + 1j*vectors[::-1]
    expected = np.block([[a, bb], [-bb, -a]]) @ vectors
    def unexpected(*args, **kwargs):
        pytest.fail('Unsupported layout reached the compiled kernel')
    monkeypatch.setattr(b, '_bse_contractions', SimpleNamespace(action=unexpected))
    np.testing.assert_allclose(b._bse_full_matmat(m, vectors, 2), expected, atol=1e-12)
    m.threads = 2
    with pytest.raises(NotImplementedError, match='row-contiguous'):
        b._bse_full_matmat(m, vectors, 2)


def test_direct_rhf_factorized_screening_and_public_threads():
    kernel = pytest.importorskip('pyqed.gw._bse_contractions', exc_type=ModuleNotFoundError)
    if not kernel.openmp_available():
        pytest.skip('OpenMP unavailable')
    from pyqed.qchem import Molecule
    mol = Molecule(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', unit='angstrom').build(eri='cd')
    mf = mol.RHF().run(tol=1e-11, verbose=0)
    assert mf.converged
    for cls in [b.BSE, b.TDA]:
        expected = cls(mf).run(nroots=3, low_rank=False).e
        actual = cls(mf, threads=4).run(nroots=3, batch_columns=3, tol=1e-9)
        assert isinstance(actual._static_screening, StaticScreening)
        np.testing.assert_allclose(actual.e, expected, atol=1e-8, rtol=0)
        assert actual.threading_info['actual_threads'] > 1


def test_automatic_batches_use_memory_not_thread_equality():
    m = SimpleNamespace(nocc=10, nso=30, threads=1, max_memory=.6)
    assert b._resolve_bse_batch(m, 'auto') == 8
    m.threads = 4
    assert b._resolve_bse_batch(m, 'auto') == 2
    m.threads = 8
    assert b._resolve_bse_batch(m, 'auto') == 1
    assert b._resolve_bse_batch(m, 16) == 16
    assert b._resolve_bse_batch(m, None) is None
    assert b._resolve_bse_batch(m, 'auto', 'arpack') is None
    small = model()
    energies, _, info = b.solve_bse(small, nroots=3, tol=1e-10, return_info=True)
    a, bb = dense_blocks(small)
    roots = np.linalg.eigvals(np.block([[a, bb], [-bb, -a]])).real
    expected = np.sort(roots[roots > 0])[:3]
    np.testing.assert_allclose(energies, expected, atol=1e-9, rtol=0)
    assert info['batch_columns'] == small.batch_columns == 8
    assert info['converged'] and np.all(energies > 0)


@pytest.mark.parametrize('half', ['both', 'x', 'y'])
def test_large_blas_products_against_dense(half):
    """Exercise BLAS-sized products, padded factor rows and zero halves."""
    kernel = pytest.importorskip("pyqed.gw._bse_contractions", exc_type=ModuleNotFoundError)
    if not kernel.openmp_available():
        pytest.skip("OpenMP unavailable")
    rng = np.random.default_rng(311)
    no, nv, rank = 20, 25, 33
    f = rng.normal(scale=.005, size=(rank, no+nv+2, no+nv+2))
    f = ((f + f.transpose(0, 2, 1))/2)[:, 1:-1, 1:-1]
    p = rng.normal(scale=.02, size=(rank, 5))
    w = StaticScreening(f, cho_factor(np.eye(rank)+p@p.T))
    energy = np.linspace(-2, 2, no+nv)
    m = SimpleNamespace(nocc=no, nso=no+nv, e_mf=energy, e_qp=None,
                        _pair_factors=f, _static_screening=w, threads=8, max_memory=50.)
    a = (2*np.einsum('Pai,Pbj->iajb', f[:, no:, :no], f[:, no:, :no])
         +np.einsum('Pij,Pab->iajb', 4*w.static_occupied(no)-f[:, :no, :no], f[:, no:, no:])).reshape(no*nv, no*nv)
    bb = (2*np.einsum('Pai,Pjb->iajb', f[:, no:, :no], f[:, :no, no:])
          +np.einsum('Pib,Paj->iajb', 4*w.static_ov(no)-f[:, :no, no:], f[:, no:, :no])).reshape(no*nv, no*nv)
    a += np.diag((energy[no:]-energy[:no,None]).ravel())
    x = rng.normal(size=(2*no*nv, 4))
    if half == 'x': x[no*nv:] = 0
    if half == 'y': x[:no*nv] = 0
    expected = np.block([[a,bb],[-bb,-a]])@x
    np.testing.assert_allclose(b._bse_full_matmat(m, x, 4), expected, atol=2e-12)
    before = m.threading_info['action_calls']
    np.testing.assert_allclose(b._bse_full_matmat(m, x.astype(complex), 4), expected, atol=2e-12)
    assert m.threading_info['action_calls']-before == 1
    np.testing.assert_allclose(b._bse_full_matmat(m, x*(1+2j), 4), expected*(1+2j), atol=3e-12)


@pytest.mark.parametrize('kind', ['a', 'b', 'full'])
def test_single_entry_ri_columns(kind):
    m = model(static=True, window=True)
    a, bb = dense_blocks(m)
    a -= np.diag((m.e_mf[m.nocc:]-m.e_mf[:m.nocc,None]).ravel())
    matrix = {'a':a,'b':bb,'full':np.block([[a,bb],[-bb,-a]])}[kind]
    x = np.eye(len(matrix))[:,::2]*2
    x[:,-1] = 0
    kernel = pytest.importorskip('pyqed.gw._bse_contractions', exc_type=ModuleNotFoundError)
    if not kernel.openmp_available(): pytest.skip('OpenMP unavailable')
    actual, _, _ = kernel.action(m._pair_factors,m._M.static_occupied(m.nocc),
        m._M.static_ov(m.nocc),np.ascontiguousarray(x),m.nocc,kind,8)
    np.testing.assert_allclose(actual,matrix@x,atol=1e-12)
