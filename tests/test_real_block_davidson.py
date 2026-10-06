"""Dense references for the shared compiled multi-root Davidson core."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
import pytest


@pytest.fixture(scope='module', params=[False, True], ids=['accelerate','portable'])
def binding(tmp_path_factory, request):
    if sys.platform != 'darwin' or not shutil.which('clang++'):
        pytest.skip('Benchmark binding requires macOS Accelerate and clang++')
    pybind11 = pytest.importorskip('pybind11')
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp('davidson')/('davidson_benchmark_binding'+sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run(['clang++', '-O3', '-std=c++17', '-shared', '-undefined', 'dynamic_lookup',
        '-fPIC', '-I'+pybind11.get_include(), '-I'+sysconfig.get_path('include'),
        '-I'+str(root/'pyqed/linalg'), str(root/'benchmarks/davidson_benchmark_binding.cpp'),
        '-framework', 'Accelerate', '-o', str(output)] +
        (['-DPYQED_DAVIDSON_PORTABLE'] if request.param else []), check=True)
    spec = importlib.util.spec_from_file_location('davidson_benchmark_binding', output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def solve(binding):
    return binding.solve


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('reuse', [False, True])
@pytest.mark.parametrize('kind', ['coupled', 'degenerate', 'full'])
def test_compiled_roots_match_dense(solve, batched, reuse, kind):
    rng = np.random.default_rng(89)
    n = 48
    matrix = rng.normal(size=(n,n))*.01
    matrix = (matrix+matrix.T)/2+np.diag(np.arange(n,dtype=float))
    k, space = 5, 24
    if kind == 'degenerate':
        q, _ = np.linalg.qr(rng.normal(size=(n,n)))
        matrix = (q*np.repeat(np.arange(n//2,dtype=float),2))@q.T
        k, space = 6, n
    elif kind == 'full':
        k, space = n, n
    values, vectors, info = solve(matrix, k, 1e-10, 100, space, batched, reuse)
    assert info['converged']
    assert info['workspace_reused'] == reuse
    assert info['basis_size'] <= space
    np.testing.assert_allclose(values, np.linalg.eigvalsh(matrix)[:k], atol=1e-9, rtol=0)
    assert np.max(np.linalg.norm(matrix@vectors-vectors*np.asarray(values),axis=0)) < 1e-10
    np.testing.assert_allclose(vectors.T@vectors, np.eye(k), atol=1e-10)


def test_partial_roots_share_one_orthogonal_space(solve):
    rng = np.random.default_rng(7)
    a = rng.normal(size=(50,50)); a=(a+a.T)/2
    values, vectors, info = solve(a, 4, 1e-14, 2, 18)
    assert not info['converged']
    np.testing.assert_allclose(vectors.T@vectors,np.eye(4),atol=1e-11)
    np.testing.assert_allclose(np.diag(vectors.T@a@vectors),values,atol=1e-11)


@pytest.mark.parametrize('kind',['scaled','dependent','near_dependent','small'])
def test_guarded_block_qr_preserves_span_or_input(binding,kind):
    rng=np.random.default_rng(591)
    a=rng.normal(size=(80,12))
    if kind=='scaled':a*=np.geomspace(1e-9,1e3,12)
    elif kind=='dependent':a[:,-1]=a[:,0]
    elif kind=='near_dependent':a[:,-1]=a[:,0]+1e-8*a[:,-1]
    elif kind=='small':a=a[:,:4].copy()
    q,accepted=binding.orthogonalize(a)
    if kind=='scaled' and binding.cholesky_available:
        assert accepted
        np.testing.assert_allclose(q.T@q,np.eye(q.shape[1]),atol=1e-12)
        normalized=a/np.linalg.norm(a,axis=0)
        np.testing.assert_allclose(q@(q.T@normalized),normalized,atol=1e-12)
    else:
        assert not accepted
        np.testing.assert_array_equal(q,a)


def test_locally_optimal_restart_on_flat_diagonal(binding):
    # Flat Jacobi preconditioning exposed loss of useful directions at restart.
    nx,ny,k=12,18,8
    tx=2*np.eye(nx)-np.eye(nx,k=1)-np.eye(nx,k=-1)
    ty=2*np.eye(ny)-np.eye(ny,k=1)-np.eye(ny,k=-1)
    a=np.kron(tx,np.eye(ny))+np.sqrt(2)*np.kron(np.eye(nx),ty)
    seed=np.linalg.qr(np.random.default_rng(2901).normal(size=(nx*ny,2*k)))[0][:,:k].copy()
    w,v,info=binding.davidson(a,k,1e-10,65,4*k,512*1024**2,seed)
    reference=np.sort((4*np.sin(np.pi*np.arange(1,nx+1)/(2*(nx+1)))[:,None]**2
        +4*np.sqrt(2)*np.sin(np.pi*np.arange(1,ny+1)/(2*(ny+1)))[None,:]**2).ravel())[:k]
    assert info['converged']
    assert info['restart_history_vectors']>0
    np.testing.assert_allclose(w,reference,atol=1e-10,rtol=0)
    assert max(np.linalg.norm(a@v-v*np.asarray(w),axis=0))<1e-10
    np.testing.assert_allclose(v.T@v,np.eye(k),atol=1e-11)


@pytest.mark.parametrize('kind', ['coupled', 'degenerate', 'full', 'partial', 'restart'])
@pytest.mark.parametrize('guess_kind', ['coordinate', 'real', 'complex'])
def test_complex_hermitian_roots(binding, kind, guess_kind):
    rng = np.random.default_rng(207)
    n, k, space = 40, 4, 20
    a = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    h = .01*(a+a.conj().T) + np.diag(np.arange(n, dtype=float))
    iterations = 150
    if kind in ('degenerate', 'full'):
        q, _ = np.linalg.qr(a)
        h = (q*np.repeat(np.arange(n//2), 2))@q.conj().T
        space = n
        if kind == 'full':
            k = n
    elif kind == 'partial':
        h = (a+a.conj().T)/2
        iterations = 2
    elif kind == 'restart':
        t = 2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1)
        phase = np.exp(1j*rng.uniform(-np.pi,np.pi,n))
        h = phase[:,None]*t*phase.conj()[None,:]
        space = 16
    guess = None
    if guess_kind != 'coordinate':
        guess = rng.normal(size=(n,k))
        if guess_kind == 'complex':
            guess = guess + 1j*rng.normal(size=(n,k))
        # Include a dependent guess to exercise rank filtering.
        if k > 1:
            guess[:,-1] = guess[:,0]
    w, v, info = binding.davidson(np.ascontiguousarray(h), k, 1e-10,
                                      iterations, space, 512*1024**2, guess)
    w = np.asarray(w)
    assert v.dtype == np.complex128
    residuals = np.linalg.norm(h@v-v*w, axis=0)
    np.testing.assert_allclose(residuals, info['residual_norms'], atol=1e-12, rtol=1e-5)
    np.testing.assert_allclose(v.conj().T@v, np.eye(k), atol=1e-10)
    np.testing.assert_allclose(np.diag(v.conj().T@h@v), w, atol=1e-10)
    if kind == 'partial':
        assert not info['converged']
    else:
        assert info['converged']
        np.testing.assert_allclose(w, np.linalg.eigvalsh(h)[:k], atol=1e-9, rtol=0)
        assert max(residuals) < 1e-10
    if kind == 'restart':
        assert info['restarts'] > 0
        assert info['restart_history_vectors'] > 0


@pytest.mark.parametrize('invalid', ['asymmetric', 'diagonal', 'nan', 'inf'])
def test_complex_input_validation(binding, invalid):
    h = np.eye(5, dtype=complex)
    if invalid == 'asymmetric':
        h[0,1] = h[1,0] = 1j
    elif invalid == 'diagonal':
        h[0,0] += 1j
    elif invalid == 'nan':
        h[0,0] = np.nan
    else:
        h[0,0] = np.inf
    with pytest.raises(ValueError, match='Hermitian'):
        binding.davidson(h, 2)


def test_complex_memory_budget_and_scalar(binding):
    h = np.eye(10, dtype=complex)
    with pytest.raises(RuntimeError, match='memory limit'):
        binding.davidson(h, 2, memory_limit=1)
    w, v, info = binding.davidson(np.array([[2+0j]]), 1)
    assert info['converged']
    np.testing.assert_allclose(w, [2.])
    np.testing.assert_allclose(v, [[1.]])
    _, _, real_info = binding.davidson(h.real.copy(), 2)
    _, _, complex_info = binding.davidson(h, 2)
    assert complex_info['estimated_peak_workspace_bytes'] == 2*real_info['estimated_peak_workspace_bytes']


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('reuse', [False, True])
def test_complex_callback_and_padded_workspace(binding, batched, reuse):
    rng = np.random.default_rng(920)
    a = rng.normal(size=(48,48)) + 1j*rng.normal(size=(48,48))
    h = .02*(a+a.conj().T)+np.diag(np.arange(48))
    w, v, info = binding.solve(h, 5, 1e-10, 100, 24, batched, reuse)
    assert info['converged']
    assert info['workspace_reused'] == reuse
    assert info['basis_size'] <= 24
    np.testing.assert_allclose(w, np.linalg.eigvalsh(h)[:5], atol=1e-10, rtol=0)
    np.testing.assert_allclose(v.conj().T@v, np.eye(5), atol=1e-11)
    assert max(np.linalg.norm(h@v-v*np.asarray(w), axis=0)) < 1e-10


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('batched', [False, True])
def test_matrix_free_binding(binding, complex_values, batched):
    rng = np.random.default_rng(219)
    a = rng.normal(size=(40,40))
    if complex_values:
        a = a + 1j*rng.normal(size=a.shape)
    h = .03*(a+a.conj().T)+np.diag(np.arange(40))
    retained = []
    def action(x):
        assert x.ndim == (2 if batched else 1)
        assert x.flags.owndata
        out = h@x
        # Retaining or mutating callback inputs must not affect solver memory.
        x[:] = 0
        retained.append(x)
        return out
    w, v, info = binding.davidson_operator(
        None if batched else action, h.diagonal().real.copy(), 4,
        1e-10, 200, 16, 512*1024**2, None, action if batched else None,
        complex_values)
    assert info['converged']
    np.testing.assert_allclose(w, np.linalg.eigvalsh(h)[:4], atol=1e-10)
    assert max(np.linalg.norm(h@v-v*np.asarray(w), axis=0)) < 1e-10
    np.testing.assert_allclose(v.conj().T@v, np.eye(4), atol=1e-10)
    assert all(np.count_nonzero(x) == 0 for x in retained)
    assert info['matmat_callback_calls'] == (len(retained) if batched else 0)
    assert info['matvec_callback_calls'] == (0 if batched else len(retained))
    assert info['matvecs'] == sum(x.shape[1] if batched else 1 for x in retained)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('bad', ['shape', 'nan', 'complex', 'exception'])
def test_matrix_free_callback_validation(binding, batched, bad):
    class CallbackFailure(Exception):
        pass
    def action(x):
        if bad == 'shape':
            return x[:-1]
        if bad == 'nan':
            return np.full_like(x, np.nan)
        if bad == 'complex':
            return x + 1j
        raise CallbackFailure('callback failure')
    error = CallbackFailure if bad == 'exception' else ValueError
    with pytest.raises(error):
        binding.davidson_operator(None if batched else action, np.arange(8.), 2,
                                  matmat=action if batched else None, complex_values=False)


@pytest.mark.parametrize('complex_values',[False,True])
def test_conditional_action_strided_blocks_and_ownership(binding,complex_values):
    rng=np.random.default_rng(345)
    def sample(shape):
        x=rng.normal(size=shape)
        return x+1j*rng.normal(size=shape) if complex_values else x
    matrices=[sample((15,15)) for _ in range(2)]
    blocks=sample((3,5,5))
    action_type=binding.ComplexConditionalAction if complex_values else binding.RealConditionalAction
    action=action_type(matrices,blocks,15)
    x=sample((15,7))
    expected=sum(a@x for a in matrices)
    expected+=(blocks@x.reshape(3,5,7)).reshape(15,7)
    np.testing.assert_allclose(action.matmat(x),expected,atol=1e-12)
    np.testing.assert_allclose(action.matmat(np.asfortranarray(x)),expected,atol=1e-12)
    saved=action.matmat(x);action.matmat(x*2)
    np.testing.assert_allclose(saved,expected,atol=1e-12)
    assert saved.flags.f_contiguous
    del matrices,blocks
    np.testing.assert_allclose(action.matmat(x),expected,atol=1e-12)
    assert action.matmat(x[:,:0]).shape==(15,0)
    with pytest.raises(ValueError):action.matmat(x[:-1])
    with pytest.raises(ValueError):action_type([],np.ones((2,3,3)),15)
    if not complex_values:
        with pytest.raises(TypeError):action.matmat(x+1j*x)


@pytest.mark.parametrize('complex_matrix', [False, True])
@pytest.mark.parametrize('gap', [0., 1e-8, .2])
def test_deflation_preserves_clustered_lowest_roots(binding, complex_matrix, gap):
    # Two exact low states converge immediately; a coupled cluster must still
    # be resolved after freezing them, including a degeneracy at the boundary.
    rng = np.random.default_rng(592)
    n, k = 64, 6
    x = rng.normal(size=(n-2, n-2))
    if complex_matrix:
        x = x + 1j*rng.normal(size=x.shape)
    q, _ = np.linalg.qr(x)
    spectrum = np.r_[gap, gap+1e-9, .4, .6, np.linspace(1., 8., n-6)]
    h = np.zeros((n, n), dtype=q.dtype)
    h[:2, :2] = np.diag([-1., 0.])
    h[2:, 2:] = (q*spectrum)@q.conj().T
    w, v, info = binding.davidson(np.ascontiguousarray(h), k, 1e-9,
                                 300, 24, 512*1024**2, None)
    w = np.asarray(w)
    assert info['converged']
    assert info['locked_roots'] >= 2
    assert info['deflated_iterations'] > 0
    np.testing.assert_allclose(w, np.linalg.eigvalsh(h)[:k], atol=1e-9, rtol=0)
    assert np.max(np.linalg.norm(h@v-v*w, axis=0)) < 1e-9
    np.testing.assert_allclose(v.conj().T@v, np.eye(k), atol=1e-10)


@pytest.mark.parametrize('complex_matrix', [False, True])
def test_deflation_unlocks_when_lower_state_is_discovered(binding, complex_matrix):
    n, k = 40, 3
    phase = np.exp(1j*np.arange(n-2)) if complex_matrix else np.ones(n-2)
    u = phase/np.sqrt(n-2)
    h = np.zeros((n, n), dtype=u.dtype)
    h[:2, :2] = np.diag([-1., 0.])
    h[2:, 2:] = 4*np.eye(n-2)-6*np.outer(u, u.conj())
    w, v, info = binding.davidson(np.ascontiguousarray(h), k, 1e-10,
                                 100, 6, 512*1024**2, None)
    w = np.asarray(w)
    assert info['converged']
    assert info['locked_roots'] == 2
    assert info['deflation_unlocks'] == 1
    np.testing.assert_allclose(w, [-2., -1., 0.], atol=1e-10, rtol=0)
    assert np.max(np.linalg.norm(h@v-v*w, axis=0)) < 1e-10
    np.testing.assert_allclose(v.conj().T@v, np.eye(k), atol=1e-11)
