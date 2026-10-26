"""Bracket-first reference and orbital OpenMP numerical agreement."""
import importlib
import numpy as np
import pytest

g = importlib.import_module('pyqed.gw.gw')


def problem():
    rng = np.random.default_rng(512)
    poles = np.r_[np.linspace(-4., -1., 40), np.linspace(.7, 4., 80)]
    residues = rng.uniform(0., .004, (8, len(poles)))
    energies = np.linspace(-.6, .5, 8)
    shifts = rng.uniform(-.1, .1, 8)
    return poles, residues, energies, shifts


def reference(poles, residues, energies, shifts, eta):
    roots, weights, residuals = [], [], []
    for residue, energy, shift in zip(residues, energies, shifts):
        correction = lambda w: np.sum(residue/(w-poles+1j*eta))+shift
        derivative = lambda w: -np.sum(residue/(w-poles+1j*eta)**2)
        root = g._continue_qp_root(correction, derivative, energy)
        roots.append(root)
        weights.append(1/(1-derivative(root).real))
        residuals.append(abs(root-energy-correction(root).real))
    return np.array([roots, weights, residuals])


def test_brackets_select_local_crossing():
    # Three positive-weight solutions: unrestricted Newton may choose any one.
    correction = lambda w: .08/(w+.15+.001j)+.12/(w-.6+.001j)
    derivative = lambda w: -.08/(w+.15+.001j)**2-.12/(w-.6+.001j)**2
    root = g._continue_qp_root(correction, derivative, .1)
    perturbed = g._continue_qp_root(lambda w: correction(w)+1e-14, derivative, .1)
    assert -.15 < root < .6
    assert root == pytest.approx(perturbed, abs=1e-10)


@pytest.mark.parametrize('eta', [0., 1e-3, .1])
@pytest.mark.parametrize('threads', [1, 2, 4, 8])
def test_orbital_solver_matches_reference(eta, threads):
    solver = pytest.importorskip('pyqed.gw._dyson', exc_type=ModuleNotFoundError)
    if threads > 1 and not solver.openmp_available():
        pytest.skip('OpenMP unavailable')
    data = problem()
    expected = reference(*data, eta)
    roots, weights, residuals, actual = solver.solve(*data, eta, threads)
    np.testing.assert_allclose(roots, expected[0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(weights, expected[1], atol=1e-9, rtol=0)
    assert np.max(residuals) <= 1e-9
    assert actual == threads


def test_orbital_solver_rejects_invalid_input():
    solver = pytest.importorskip('pyqed.gw._dyson', exc_type=ModuleNotFoundError)
    data = problem()
    with pytest.raises(ValueError):
        solver.solve(*data, np.nan, 1)
    data[1][0, 0] = -1.
    with pytest.raises(ValueError):
        solver.solve(*data, .001, 1)


def test_public_factorized_solver_threads_and_memory_batching(monkeypatch):
    solver = pytest.importorskip('pyqed.gw._dyson', exc_type=ModuleNotFoundError)
    if not solver.openmp_available():
        pytest.skip('OpenMP unavailable')
    from types import SimpleNamespace
    rng = np.random.default_rng(852)
    factors = rng.normal(scale=.04, size=(6, 5, 5))
    factors = (factors+factors.transpose(0, 2, 1))/2
    def model(threads, memory):
        return SimpleNamespace(nso=10, nocc=4, e_mf=np.repeat([-1., -.5, .2, .4, .7], 2),
            _qp_energy_so=None, _M=None, _pair_factors=factors, eri=None,
            _sigma_x_matrix=None, _charge_screening=None, mo_coeff=np.eye(5), eta=.001,
            v_mf=np.zeros((10, 10)), max_memory=memory, threads=threads)
    one = model(1, 4.)
    poles, vectors = g.rpa(one)
    monkeypatch.setattr(g.FactorizedCouplings, '__array__',
                        lambda *a, **kw: pytest.fail('full coupling tensor materialized'))
    expected = g._solve_qp_energies(one, poles, vectors)
    many = model(4, 4.)
    actual = g._solve_qp_energies(many, poles, vectors)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    assert many.threading_info['actual_threads'] == 4
    small = model(4, .0001)
    bounded = g._solve_qp_energies(small, poles, vectors)
    np.testing.assert_allclose(bounded, expected, atol=1e-10, rtol=0)
    assert small.threading_info['batch_orbitals'] == 1
    assert small.threading_info['actual_threads'] == 1


@pytest.mark.parametrize('threads', [0, -1, 1.5, True])
def test_public_thread_validation(threads):
    with pytest.raises(ValueError, match='positive integer'):
        g.GW(None, threads=threads, freq_int='exact')


@pytest.mark.parametrize('contour', ['cd', 'CD', 'contour-deformation'])
def test_missing_openmp_fails_before_molecular_preparation(monkeypatch, contour):
    monkeypatch.setattr(g, '_dyson', None)
    with pytest.raises(RuntimeError, match='with OpenMP'):
        g.GW(None, threads=2, freq_int='exact')
    c = importlib.import_module('pyqed.gw.contour')
    monkeypatch.setattr(c, '_contour_contractions', None)
    with pytest.raises(RuntimeError, match='contour contractions'):
        g.GW(None, threads=2, freq_int=contour)


@pytest.mark.parametrize('torch_first', [False, True])
def test_torch_and_dyson_share_openmp_runtime(torch_first):
    """Exercise both runtimes in a child process so an abort is a test failure."""
    import importlib.util
    import os
    import subprocess
    import sys
    if importlib.util.find_spec('torch') is None:
        pytest.skip('PyTorch is not installed')
    pytest.importorskip('pyqed.gw._dyson', exc_type=ModuleNotFoundError)
    torch_code = "import torch; torch.set_num_threads(2); torch.mm(torch.ones(128,128), torch.ones(128,128))"
    dyson_code = '''import numpy as np
from pyqed.gw import _dyson
poles = np.array([-2., 2.])
residues = np.full((2, 2), .01)
result = _dyson.solve(poles, residues, np.array([-.5, .5]), np.zeros(2), .001, 2 if _dyson.openmp_available() else 1)
assert np.all(np.isfinite(result[0]))'''
    chunks = [torch_code, dyson_code] if torch_first else [dyson_code, torch_code]
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    result = subprocess.run([sys.executable, '-c', '\n'.join(chunks)],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
